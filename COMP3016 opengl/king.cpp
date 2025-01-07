#include <iostream>
#include <string>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <cmath>
#include <cstring>
#include <array>

struct Position {
    float x, y, z;
};

struct MovingPiece {
    Position current;
    Position target;
    bool isMoving;
    float moveSpeed;
    bool isWhite;
};

std::vector<MovingPiece> piecePositions;
const float MOVE_SPEED = 2.0f; 

void handleSquareSelection(int gridX, int gridZ);

enum class PieceType {
    EMPTY,
    PAWN_BLACK, PAWN_WHITE,
    ROOK_BLACK, ROOK_WHITE,
    KNIGHT_BLACK, KNIGHT_WHITE,
    BISHOP_BLACK, BISHOP_WHITE,
    QUEEN_BLACK, QUEEN_WHITE,
    KING_BLACK, KING_WHITE
};

// ChessBoard class definition
class ChessBoard {
public:
    std::array<std::array<PieceType, 8>, 8> board;

    ChessBoard() {
        // Initialize black pieces (row 0)
        board[0][0] = PieceType::ROOK_BLACK;
        board[0][1] = PieceType::KNIGHT_BLACK;
        board[0][2] = PieceType::BISHOP_BLACK;
        board[0][3] = PieceType::KING_BLACK;
        board[0][4] = PieceType::QUEEN_BLACK;
        board[0][5] = PieceType::BISHOP_BLACK;
        board[0][6] = PieceType::KNIGHT_BLACK;
        board[0][7] = PieceType::ROOK_BLACK;

        // Black pawns (row 1)
        for (int i = 0; i < 8; i++) {
            board[1][i] = PieceType::PAWN_BLACK;
        }

        // Empty squares (rows 2-5)
        for (int row = 2; row < 6; row++) {
            for (int col = 0; col < 8; col++) {
                board[row][col] = PieceType::EMPTY;
            }
        }

        // White pawns (row 6)
        for (int i = 0; i < 8; i++) {
            board[6][i] = PieceType::PAWN_WHITE;
        }

        // White pieces (row 7)
        board[7][0] = PieceType::ROOK_WHITE;
        board[7][1] = PieceType::KNIGHT_WHITE;
        board[7][2] = PieceType::BISHOP_WHITE;
        board[7][3] = PieceType::KING_WHITE;
        board[7][4] = PieceType::QUEEN_WHITE;
        board[7][5] = PieceType::BISHOP_WHITE;
        board[7][6] = PieceType::KNIGHT_WHITE;
        board[7][7] = PieceType::ROOK_WHITE;
    }

    void printBoard() {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                char piece;
                switch (board[i][j]) {
                    case PieceType::EMPTY: piece = '.'; break;
                    case PieceType::PAWN_BLACK: piece = 'P'; break;
                    case PieceType::PAWN_WHITE: piece = 'p'; break;
                    case PieceType::ROOK_BLACK: piece = 'R'; break;
                    case PieceType::ROOK_WHITE: piece = 'r'; break;
                    case PieceType::KNIGHT_BLACK: piece = 'N'; break;
                    case PieceType::KNIGHT_WHITE: piece = 'n'; break;
                    case PieceType::BISHOP_BLACK: piece = 'B'; break;
                    case PieceType::BISHOP_WHITE: piece = 'b'; break;
                    case PieceType::QUEEN_BLACK: piece = 'Q'; break;
                    case PieceType::QUEEN_WHITE: piece = 'q'; break;
                    case PieceType::KING_BLACK: piece = 'K'; break;
                    case PieceType::KING_WHITE: piece = 'k'; break;
                }
                std::cout << piece << " ";
            }
            std::cout << std::endl;
        }
    }
};

// Declare matrices globally or pass them as parameters
float viewMatrix[16];
float projectionMatrix[16];


// Function to print matrix
void printMatrix(const float* matrix, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << matrix[i * 4 + j] << " ";
        }
        std::cout << "\n";
    }
}

// Function to load a 3D model
void loadModel(const std::string& modelName, std::vector<float>& vertices, std::vector<unsigned int>& indices) {
    std::cout << "Loading model: " << modelName << std::endl;
    std::string filePath = "assets/3D/Peices/" + modelName;
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filePath, aiProcess_Triangulate | aiProcess_FlipUVs);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        return;
    }

    // Process all meshes
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        aiMesh* mesh = scene->mMeshes[m];
        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            vertices.push_back(mesh->mVertices[i].x);
            vertices.push_back(mesh->mVertices[i].y);
            vertices.push_back(mesh->mVertices[i].z);
            vertices.push_back(1.0f); // Color placeholder
            vertices.push_back(1.0f);
            vertices.push_back(1.0f);
        }
        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++) {
                indices.push_back(face.mIndices[j]);
            }
        }
    }
    std::cout << "Loaded model: " << modelName << " with " << vertices.size() / 6 << " vertices and " << indices.size() << " indices." << std::endl;
}

// Function to check OpenGL errors
void checkOpenGLError(const std::string& location) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "OpenGL error at " << location << ": " << err << std::endl;
    }
}

// Function to set up the chessboard
void setupChessboard(std::vector<float>& vertices, std::vector<unsigned int>& indices) {
    float size = 100.0f;
    float height = 10.0f; // Thickness of the board
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            float x = i * size;
            float z = j * size;
            float colorR = (i + j) % 2 == 0 ? 0.8f : 0.0f; // Cream for even squares
            float colorG = (i + j) % 2 == 0 ? 0.8f : 0.5f; // Green for odd squares
            float colorB = (i + j) % 2 == 0 ? 0.6f : 0.0f; 

            // Bottom face
            vertices.insert(vertices.end(), {
                x, 0.0f, z, colorR, colorG, colorB,
                x + size, 0.0f, z, colorR, colorG, colorB,
                x + size, 0.0f, z + size, colorR, colorG, colorB,
                x, 0.0f, z + size, colorR, colorG, colorB
            });

            // Top face (thickness)
            vertices.insert(vertices.end(), {
                x, height, z, colorR, colorG, colorB,
                x + size, height, z, colorR, colorG, colorB,
                x + size, height, z + size, colorR, colorG, colorB,
                x, height, z + size, colorR, colorG, colorB
            });

            unsigned int offset = (i * 8 + j) * 4;
            indices.insert(indices.end(), {
                offset, offset + 1, offset + 2,
                offset + 2, offset + 3, offset
            });

            // Indices for the top face
            unsigned int topOffset = (i * 8 + j + 64) * 4; // Offset for the top face
            indices.insert(indices.end(), {
                topOffset, topOffset + 1, topOffset + 2,
                topOffset + 2, topOffset + 3, topOffset
            });
        }
    }
}

// Function to compile shaders
GLuint compileShaders() {
    std::cout << "Compiling shaders..." << std::endl;
    const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;
        out vec3 ourColor;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            ourColor = aColor;
        }
    )";

    const char* fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;
        in vec3 ourColor;
        void main() {
            FragColor = vec4(ourColor, 1.0);
        }
    )";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    GLint success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    std::cout << "Shaders compiled successfully." << std::endl;
    return shaderProgram;
}

// Setup VAO
GLuint setupVAO(const std::vector<float>& vertices, const std::vector<unsigned int>& indices) {
    std::cout << "Setting up VAO..." << std::endl;
    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return VAO;
}

// Function to set identity matrix
void setIdentityMatrix(float* matrix) {
    std::memset(matrix, 0, 16 * sizeof(float));
    matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1.0f;
}

// Function to translate matrix
void translateMatrix(float* matrix, float x, float y, float z) {
    matrix[12] += x;
    matrix[13] += y;
    matrix[14] += z;
}

// Function to create perspective matrix
void perspectiveMatrix(float* matrix, float fov, float aspect, float near, float far) {
    float tanHalfFov = tan(fov / 2.0f);
    matrix[0] = 1.0f / (aspect * tanHalfFov);
    matrix[1] = 0.0f;
    matrix[2] = 0.0f;
    matrix[3] = 0.0f;

    matrix[4] = 0.0f;
    matrix[5] = 1.0f / tanHalfFov;
    matrix[6] = 0.0f;
    matrix[7] = 0.0f;

    matrix[8] = 0.0f;
    matrix[9] = 0.0f;
    matrix[10] = -(far + near) / (far - near);
    matrix[11] = -1.0f;

    matrix[12] = 0.0f;
    matrix[13] = 0.0f;
    matrix[14] = -(2.0f * far * near) / (far - near);
    matrix[15] = 0.0f;
}

// Function to rotate matrix around the x-axis
void rotateMatrixX(float* matrix, float angle) {
    float rad = angle * (3.14159265359f / 180.0f); // Convert to radians
    float cosA = cos(rad);
    float sinA = sin(rad);

    matrix[5] = cosA;
    matrix[6] = -sinA;
    matrix[9] = sinA;
    matrix[10] = cosA;
}

// Function to rotate matrix around the z-axis
void rotateMatrixZ(float* matrix, float angle) {
    float rad = angle * (3.14159265359f / 180.0f); // Convert to radians
    float cosA = cos(rad);
    float sinA = sin(rad);

    float temp0 = matrix[0] * cosA - matrix[1] * sinA;
    float temp1 = matrix[0] * sinA + matrix[1] * cosA;
    float temp4 = matrix[4] * cosA - matrix[5] * sinA;
    float temp5 = matrix[4] * sinA + matrix[5] * cosA;
    float temp8 = matrix[8] * cosA - matrix[9] * sinA;
    float temp9 = matrix[8] * sinA + matrix[9] * cosA;

    matrix[0] = temp0;
    matrix[1] = temp1;
    matrix[4] = temp4;
    matrix[5] = temp5;
    matrix[8] = temp8;
    matrix[9] = temp9;
}

// Function to scale matrix
void scaleMatrix(float* matrix, float scale) {
    matrix[0] *= scale;  // Scale x-axis
    matrix[5] *= scale;  // Scale y-axis
    matrix[10] *= scale; // Scale z-axis
}

// Function to rotate matrix around the y-axis
void rotateMatrixY(float* matrix, float angle) {
    float rad = angle * (3.14159265359f / 180.0f); // Convert to radians
    float cosA = cos(rad);
    float sinA = sin(rad);

    matrix[0] = cosA;
    matrix[2] = sinA;
    matrix[8] = -sinA;
    matrix[10] = cosA;
}

// Function to render a piece
void renderPiece(GLuint shaderProgram, GLuint VAO, const float* modelMatrix, int indexCount) {
    GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, modelMatrix);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

// Function to render a piece at a specific position with scaling and rotation
void renderPieceAt(GLuint shaderProgram, GLuint VAO, float x, float y, float z, int indexCount, float scale, float rotationAngleX, float rotationAngleY, float rotationAngleZ, bool isSelected) {
    float model[16];
    setIdentityMatrix(model);
    scaleMatrix(model, scale); // Apply uniform scaling first
    rotateMatrixX(model, rotationAngleX); // Rotate around x-axis
    rotateMatrixY(model, rotationAngleY); // Rotate around y-axis
    rotateMatrixZ(model, rotationAngleZ); // Rotate around z-axis
    translateMatrix(model, x, y, z); // Apply translation last

    // Change color if selected
    if (isSelected) {
        glUniform3f(glGetUniformLocation(shaderProgram, "overrideColor"), 1.0f, 0.0f, 0.0f); // Red color for selected
    } else {
        glUniform3f(glGetUniformLocation(shaderProgram, "overrideColor"), 1.0f, 1.0f, 1.0f); // Default color
    }

    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, model);
    renderPiece(shaderProgram, VAO, model, indexCount);
}

// Initialize OpenGL
GLFWwindow* initOpenGL() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return nullptr;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Get the primary monitor
    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(primaryMonitor);

    // Create a full-screen window
    GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, "3D Chess Board", primaryMonitor, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return nullptr;
    }
    glEnable(GL_DEPTH_TEST);
    return window;
}

/*GLFWwindow* initOpenGL() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return nullptr;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a windowed mode window
    int windowWidth = 800;
    int windowHeight = 600;
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "3D Chess Board", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return nullptr;
    }
    glEnable(GL_DEPTH_TEST);
    return window;
}*/

void setBlackPieceColor(std::vector<float>& vertices) {
    for (size_t i = 0; i < vertices.size(); i += 6) {
        // Set RGB to black for each vertex
        vertices[i + 3] = 0.0f; // R
        vertices[i + 4] = 0.0f; // G
        vertices[i + 5] = 0.0f; // B
    }
}

// Structure to represent a piece
struct Piece {
    float x, y, z; // Position
    float size;    // Size of the piece (radius for bounding sphere)
    bool isSelected; // Selection state

    // Constructor
    Piece(float x, float y, float z, float size, bool isSelected)
        : x(x), y(y), z(z), size(size), isSelected(isSelected) {}
};

// List of pieces
std::vector<Piece> pieces;

// Function to check if a point is within a piece's bounding box
bool isPointInPiece(float* worldCoords, const Piece& piece) {
    return std::abs(worldCoords[0] - piece.x) < piece.size &&
           std::abs(worldCoords[1] - piece.y) < piece.size &&
           std::abs(worldCoords[2] - piece.z) < piece.size;
}

// Function to convert screen coordinates to world coordinates


// Function to multiply a 4x4 matrix with a 4x1 vector
void multiplyMatrixVector(const float* matrix, const float* vector, float* result) {
    for (int i = 0; i < 4; ++i) {
        result[i] = 0.0f;
        for (int j = 0; j < 4; ++j) {
            result[i] += matrix[i * 4 + j] * vector[j];
        }
    }
}

// Function to check if a matrix is a zero matrix
bool isZeroMatrix(const float* matrix) {
    for (int i = 0; i < 16; ++i) {
        if (matrix[i] != 0.0f) return false;
    }
    return true;
}

// Function to invert a matrix
bool invertMatrix(const float* m, float* invOut) {
    if (isZeroMatrix(m)) {
        std::cerr << "Matrix is a zero matrix, cannot invert.\n";
        return false;
    }
    // Calculate the determinant of the matrix
    float det = m[0] * (m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10]) - 
                m[4] * (m[1] * m[10] * m[15] - m[1] * m[11] * m[14] - m[9] * m[2] * m[15] + m[9] * m[3] * m[14] + m[13] * m[2] * m[11] - m[13] * m[3] * m[10]) + 
                m[8] * (m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6]) - 
                m[12] * (m[1] * m[6] * m[11] - m[1] * m[7] * m[10] - m[5] * m[2] * m[11] + m[5] * m[3] * m[10] + m[9] * m[2] * m[7] - m[9] * m[3] * m[6]);
    if (det == 0) {
        std::cerr << "Matrix is singular, cannot invert.\n";
        return false;
    }
    // Calculate the inverse of the matrix
    invOut[0] = (m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10]) / det;
    invOut[1] = -(m[1] * m[10] * m[15] - m[1] * m[11] * m[14] - m[9] * m[2] * m[15] + m[9] * m[3] * m[14] + m[13] * m[2] * m[11] - m[13] * m[3] * m[10]) / det;
    invOut[2] = (m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6]) / det;
    invOut[3] = -(m[1] * m[6] * m[11] - m[1] * m[7] * m[10] - m[5] * m[2] * m[11] + m[5] * m[3] * m[10] + m[9] * m[2] * m[7] - m[9] * m[3] * m[6]) / det;
    invOut[4] = -(m[4] * m[10] * m[15] - m[4] * m[11] * m[14] - m[8] * m[6] * m[15] + m[8] * m[7] * m[14] + m[12] * m[6] * m[11] - m[12] * m[7] * m[10]) / det;
    invOut[5] = (m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10]) / det;
    invOut[6] = -(m[0] * m[6] * m[15] - m[0] * m[7] * m[14] - m[4] * m[2] * m[15] + m[4] * m[3] * m[14] + m[12] * m[2] * m[7] - m[12] * m[3] * m[6]) / det;
    invOut[7] = (m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6]) / det;
    invOut[8] = (m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9]) / det;
    invOut[9] = -(m[0] * m[9] * m[15] - m[0] * m[11] * m[13] - m[8] * m[1] * m[15] + m[8] * m[3] * m[13] + m[12] * m[1] * m[11] - m[12] * m[3] * m[9]) / det;
    invOut[10] = (m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5]) / det;
    invOut[11] = -(m[0] * m[5] * m[11] - m[0] * m[7] * m[9] - m[4] * m[1] * m[11] + m[4] * m[3] * m[9] + m[8] * m[1] * m[7] - m[8] * m[3] * m[5]) / det;
    invOut[12] = -(m[4] * m[9] * m[14] - m[4] * m[10] * m[13] - m[8] * m[5] * m[14] + m[8] * m[6] * m[13] + m[12] * m[5] * m[10] - m[12] * m[6] * m[9]) / det;
    invOut[13] = (m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9]) / det;

    float inv[16];
    //det = 0;
    int i;

    // Print the matrix before attempting inversion
    printMatrix(m, "Matrix to Invert");



    if (det == 0) {
        std::cerr << "Matrix is singular, cannot invert\n";
        return false;
    }

    det = 1.0f / det;

    for (i = 0; i < 13; i++)
        inv[i] = invOut[i];// det;

    // Debugging: Print the inverted matrix
    printMatrix(invOut, "Inverted Matrix");

    return true;
}

// Function to move a selected piece
void moveSelectedPiece(float newX, float newY, float newZ) {
    for (auto& piece : pieces) {
        if (piece.isSelected) {
            // Update piece position
            piece.x = newX;
            piece.y = newY;
            piece.z = newZ;
            piece.isSelected = false; // Deselect after moving
            break;
        }
    }
}

// Structure to represent a ray
struct Ray {
    float origin[3];
    float direction[3];
};

// Function to normalize a vector
void normalize(float* v) {
    float length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (length != 0) {
        v[0] /= length;
        v[1] /= length;
        v[2] /= length;
    }
}

// Function to check ray-sphere intersection
bool rayIntersectsSphere(const Ray& ray, const Piece& piece) {
    float oc[3] = { ray.origin[0] - piece.x, ray.origin[1] - piece.y, ray.origin[2] - piece.z };
    float b = 2.0f * (oc[0] * ray.direction[0] + oc[1] * ray.direction[1] + oc[2] * ray.direction[2]);
    float c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - piece.size * piece.size;
    float discriminant = b * b - 4 * c;

    // Debugging: Print intersection details
    std::cout << "Checking intersection with piece at (" << piece.x << ", " << piece.y << ", " << piece.z << ")\n";
    std::cout << "Ray Origin: (" << ray.origin[0] << ", " << ray.origin[1] << ", " << ray.origin[2] << ")\n";
    std::cout << "Ray Direction: (" << ray.direction[0] << ", " << ray.direction[1] << ", " << ray.direction[2] << ")\n";
    std::cout << "Discriminant: " << discriminant << "\n";

    return (discriminant >= 0); // Intersection occurs if discriminant is non-negative
}

// Function to create a ray from the camera through the mouse position
Ray createRayFromMouse(GLFWwindow* window, double mouseX, double mouseY, float* viewMatrix, float* projectionMatrix) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // Normalize screen coordinates
    float xNDC = (2.0f * mouseX) / width - 1.0f;
    float yNDC = 1.0f - (2.0f * mouseY) / height;

    // Clip coordinates
    float clipCoords[4] = { xNDC, yNDC, -1.0f, 1.0f };

    // Eye coordinates
    float inverseProjection[16];
    if (!invertMatrix(projectionMatrix, inverseProjection)) {
        std::cerr << "Failed to invert projection matrix\n";
        return Ray(); // Return a default ray
    }
    float eyeCoords[4];
    multiplyMatrixVector(inverseProjection, clipCoords, eyeCoords);
    eyeCoords[2] = -1.0f; // Forward direction
    eyeCoords[3] = 0.0f;

    // World coordinates
    float inverseView[16];
    if (!invertMatrix(viewMatrix, inverseView)) {
        std::cerr << "Failed to invert view matrix\n";
        return Ray(); // Return a default ray
    }
    float rayWorld[4];
    multiplyMatrixVector(inverseView, eyeCoords, rayWorld);

    // Create ray
    Ray ray;
    ray.origin[0] = 0.0f; // Camera position
    ray.origin[1] = 0.0f;
    ray.origin[2] = 0.0f;
    ray.direction[0] = rayWorld[0];
    ray.direction[1] = rayWorld[1];
    ray.direction[2] = rayWorld[2];
    normalize(ray.direction);

    // Debugging: Print the ray direction
    std::cout << "Ray Direction: (" << ray.direction[0] << ", " << ray.direction[1] << ", " << ray.direction[2] << ")\n";

    return ray;
}

// Function to map mouse click to board coordinates
void mapClickToBoard(GLFWwindow* window, double mouseX, double mouseY, float& boardX, float& boardZ) {
    // Define the board boundaries 
    const float BOARD_LEFT = 700.0f;
    const float BOARD_RIGHT = 1230.0f; 
    const float BOARD_TOP = 677.0f;
    const float BOARD_BOTTOM = 1100.0f;

    // Check if click is within board boundaries
    if (mouseX < BOARD_LEFT || mouseX > BOARD_RIGHT || 
        mouseY < BOARD_TOP || mouseY > BOARD_BOTTOM) {
        std::cout << "Click outside board boundaries\n";
        boardX = -1;
        boardZ = -1;
        return;
    }

    // Calculate normalized position within board (0 to 1)
    float normalizedX = (mouseX - BOARD_LEFT) / (BOARD_RIGHT - BOARD_LEFT);
    float normalizedY = (mouseY - BOARD_TOP) / (BOARD_BOTTOM - BOARD_TOP);

    // Convert to board coordinates with adjusted scaling
    boardX = normalizedX * 800.0f;  // Scale directly to board size
    boardZ = normalizedY * 800.0f + 100.0f;

    // Adjust to center the coordinates on the grid
    boardX = std::floor(boardX / 100.0f) * 100.0f;
    boardZ = std::floor((boardZ - 50.0f) / 100.0f) * 100.0f;

    // Ensure coordinates stay within valid range
    boardX = std::max(0.0f, std::min(boardX, 700.0f));
    boardZ = std::max(0.0f, std::min(boardZ, 700.0f));

    std::cout << "Screen coords: (" << mouseX << ", " << mouseY << ")\n";
    std::cout << "Normalized coords: (" << normalizedX << ", " << normalizedY << ")\n";
    std::cout << "Board coords: (" << boardX << ", " << boardZ << ")\n";
}

std::pair<int, int> getGridSquare(float boardX, float boardZ) {
    // Define segment boundaries for X and Z
    const float xSegments[9] = {
        0.0f,   // Left edge
        100.0f, // 1st vertical line
        200.0f, // 2nd vertical line
        300.0f, // 3rd vertical line
        400.0f, // Center vertical
        500.0f, // 5th vertical line
        600.0f, // 6th vertical line
        700.0f, // 7th vertical line
        800.0f  // Right edge
    };

    const float zSegments[9] = {
        0.0f,   // Top edge
        100.0f, // 1st horizontal line
        200.0f, // 2nd horizontal line
        300.0f, // 3rd horizontal line
        400.0f, // Center horizontal
        500.0f, // 5th horizontal line
        600.0f, // 6th horizontal line
        700.0f, // 7th horizontal line
        800.0f  // Bottom edge
    };

    // Find the grid position by checking which segment contains the coordinates
    int gridX = 0;
    int gridZ = 0;

    // Find X grid position
    for (int i = 0; i < 8; i++) {
        if (boardX >= xSegments[i] && boardX < xSegments[i + 1]) {
            gridX = i;
            break;
        }
    }

    // Find Z grid position
    for (int i = 0; i < 8; i++) {
        if (boardZ >= zSegments[i] && boardZ < zSegments[i + 1]) {
            gridZ = i;
            break;
        }
    }

    // Ensure coordinates are within bounds
    gridX = std::max(0, std::min(7, gridX));
    gridZ = std::max(0, std::min(7, gridZ));
    
    std::cout << "Grid position: (" << gridX << ", " << gridZ << ")\n";
    return {gridX, gridZ};
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        
        float boardX, boardZ;
        mapClickToBoard(window, mouseX, mouseY, boardX, boardZ);
        
        // Only process if click was within board boundaries
        if (boardX >= 0 && boardZ >= 0) {
            auto [gridX, gridZ] = getGridSquare(boardX, boardZ);
            handleSquareSelection(gridX, gridZ);
        }
    }
}

void updatePieceColor(std::vector<float>& vertices, bool isSelected) {
    for (size_t i = 0; i < vertices.size(); i += 6) {
        if (isSelected) {
            vertices[i + 3] = 1.0f; // Highlight with red color
            vertices[i + 4] = 0.0f;
            vertices[i + 5] = 0.0f;
        } else {
            vertices[i + 3] = 0.0f; // Default color
            vertices[i + 4] = 0.0f;
            vertices[i + 5] = 0.0f;
        }
    }
}

void logPiecePositions() {
    for (const auto& piece : pieces) {
        std::cout << "Piece Position: (" << piece.x << ", " << piece.y << ", " << piece.z << ")\n";
    }
}

// Function to set up a view matrix manually
void setupViewMatrix(float* viewMatrix, const float* cameraPos, const float* target, const float* up) {
    float forward[3], right[3], upVector[3];

    // Calculate forward vector
    forward[0] = target[0] - cameraPos[0];
    forward[1] = target[1] - cameraPos[1];
    forward[2] = target[2] - cameraPos[2];
    normalize(forward);

    // Calculate right vector
    right[0] = forward[1] * up[2] - forward[2] * up[1];
    right[1] = forward[2] * up[0] - forward[0] * up[2];
    right[2] = forward[0] * up[1] - forward[1] * up[0];
    normalize(right);

    // Calculate up vector
    upVector[0] = right[1] * forward[2] - right[2] * forward[1];
    upVector[1] = right[2] * forward[0] - right[0] * forward[2];
    upVector[2] = right[0] * forward[1] - right[1] * forward[0];

    // Set up view matrix
    viewMatrix[0] = right[0];
    viewMatrix[1] = upVector[0];
    viewMatrix[2] = -forward[0];
    viewMatrix[3] = 0.0f;

    viewMatrix[4] = right[1];
    viewMatrix[5] = upVector[1];
    viewMatrix[6] = -forward[1];
    viewMatrix[7] = 0.0f;

    viewMatrix[8] = right[2];
    viewMatrix[9] = upVector[2];
    viewMatrix[10] = -forward[2];
    viewMatrix[11] = 0.0f;

    viewMatrix[12] = -(right[0] * cameraPos[0] + right[1] * cameraPos[1] + right[2] * cameraPos[2]);
    viewMatrix[13] = -(upVector[0] * cameraPos[0] + upVector[1] * cameraPos[1] + upVector[2] * cameraPos[2]);
    viewMatrix[14] = forward[0] * cameraPos[0] + forward[1] * cameraPos[1] + forward[2] * cameraPos[2];
    viewMatrix[15] = 1.0f;

    // Debugging: Print the view matrix
    printMatrix(viewMatrix, "View Matrix");
}

void multiplyMatrices(const float* a, const float* b, float* result) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i * 4 + j] = 0.0f;
            for (int k = 0; k < 4; ++k) {
                result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
            }
        }
    }
}


void screenToWorld(GLFWwindow* window, double x, double y, float* viewMatrix, float* projectionMatrix, float* worldCoords) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // Normalize screen coordinates
    float xNDC = (2.0f * x) / width - 1.0f;
    float yNDC = 1.0f - (2.0f * y) / height;

    // Create a 4D point in clip space
    float clipCoords[4] = { xNDC, yNDC, -1.0f, 1.0f };

    // Invert the projection matrix
    float invProjection[16];
    if (!invertMatrix(projectionMatrix, invProjection)) {
        std::cerr << "Failed to invert projection matrix" << std::endl;
        return;
    }

    // Transform clip coordinates to eye coordinates
    float eyeCoords[4];
    multiplyMatrixVector(invProjection, clipCoords, eyeCoords);
    eyeCoords[2] = -1.0f;
    eyeCoords[3] = 0.0f;

    // Invert the view matrix
    float invView[16];
    if (!invertMatrix(viewMatrix, invView)) {
        std::cerr << "Failed to invert view matrix" << std::endl;
        return;
    }

    // Transform eye coordinates to world coordinates
    multiplyMatrixVector(invView, eyeCoords, worldCoords);
}

void drawDebugGrid(GLuint shaderProgram) {
    // Create vertices for grid lines
    std::vector<float> gridVertices;
    std::vector<unsigned int> gridIndices;
    
    // Create grid lines
    for (int i = 0; i <= 8; i++) {
        float linePos = i * 100.0f;
        
        // Vertical lines
        gridVertices.insert(gridVertices.end(), {
            linePos, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,      // Start point (red)
            linePos, 0.0f, 800.0f, 1.0f, 0.0f, 0.0f     // End point (red)
        });
        
        // Horizontal lines
        gridVertices.insert(gridVertices.end(), {
            0.0f, 0.0f, linePos, 1.0f, 0.0f, 0.0f,      // Start point (red)
            800.0f, 0.0f, linePos, 1.0f, 0.0f, 0.0f     // End point (red)
        });
    }
    
    // Create indices for lines
    for (unsigned int i = 0; i < gridVertices.size() / 6; i++) {
        gridIndices.push_back(i);
    }
    
    // Create VAO and VBO for grid
    GLuint gridVAO, gridVBO, gridEBO;
    glGenVertexArrays(1, &gridVAO);
    glGenBuffers(1, &gridVBO);
    glGenBuffers(1, &gridEBO);
    
    glBindVertexArray(gridVAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glBufferData(GL_ARRAY_BUFFER, gridVertices.size() * sizeof(float), gridVertices.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gridEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, gridIndices.size() * sizeof(unsigned int), gridIndices.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Draw the grid
    float model[16];
    setIdentityMatrix(model);
    GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, model);
    
    glBindVertexArray(gridVAO);
    glDrawElements(GL_LINES, gridIndices.size(), GL_UNSIGNED_INT, 0);
    
    // Cleanup
    glDeleteVertexArrays(1, &gridVAO);
    glDeleteBuffers(1, &gridVBO);
    glDeleteBuffers(1, &gridEBO);
}

ChessBoard chessBoard;

// Add this structure to store piece information
struct SelectedPiece {
    bool isSelected = false;
    int x = -1;
    int z = -1;
    bool isWhite = false;
};

// Global variable to track selected piece
SelectedPiece selectedPiece;

// Function to handle piece selection and movement
void handleSquareSelection(int gridX, int gridZ) {
    // Debug print the board state
    std::cout << "Checking position (" << gridX << ", " << gridZ << ") on board:\n";
    chessBoard.printBoard();

    // Get the piece at the clicked position
    PieceType clickedPiece = chessBoard.board[gridZ][gridX];
    
    // Debug print what was found
    std::cout << "Found piece type: " << static_cast<int>(clickedPiece) << " at position\n";
    
    if (clickedPiece != PieceType::EMPTY) {
        if (!selectedPiece.isSelected) {
            // Select this piece
            selectedPiece.isSelected = true;
            selectedPiece.x = gridX;
            selectedPiece.z = gridZ;
            selectedPiece.isWhite = (gridZ < 4);
            std::cout << "Selected " << (selectedPiece.isWhite ? "white" : "black") 
                     << " piece at: (" << gridX << ", " << gridZ << ")\n";
        } else {
            // Check if clicking on same color piece
            bool newSquareIsWhite = (gridZ < 4);
            if (newSquareIsWhite == selectedPiece.isWhite) {
                // Change selection to this piece
                selectedPiece.x = gridX;
                selectedPiece.z = gridZ;
                std::cout << "Changed selection to piece at: (" << gridX << ", " << gridZ << ")\n";
            } else {
                // Attempt to capture opposite color piece
                // (Add capture logic here)
                std::cout << "Attempting to capture piece at: (" << gridX << ", " << gridZ << ")\n";
            }
        }
    } else if (selectedPiece.isSelected) {
        std::cout << "Moving piece from (" << selectedPiece.x << ", " << selectedPiece.z 
                  << ") to (" << gridX << ", " << gridZ << ")\n";
        
        // Update board array
        chessBoard.board[gridZ][gridX] = chessBoard.board[selectedPiece.z][selectedPiece.x];
        chessBoard.board[selectedPiece.z][selectedPiece.x] = PieceType::EMPTY;

        // Update piece movement
        for (auto& piece : piecePositions) {
            float sourceX = selectedPiece.x * 100.0f + 50.0f;
            float sourceZ = selectedPiece.z * 100.0f + 50.0f;
            
            std::cout << "Checking piece at (" << piece.current.x << ", " << piece.current.z 
                      << ") against source (" << sourceX << ", " << sourceZ << ")\n";
                      
            if (piece.current.x == sourceX && piece.current.z == sourceZ) {
                piece.target = {
                    gridX * 100.0f + 50.0f,
                    piece.current.y,
                    gridZ * 100.0f + 50.0f
                };
                piece.isMoving = true;
                std::cout << "Found matching piece, setting target to (" 
                          << piece.target.x << ", " << piece.target.z << ")\n";
                break;
            }
        }

        selectedPiece.isSelected = false;
    }
}

// Add this function to initialize piece positions
void initializePiecePositions() {
    piecePositions.clear();
    std::cout << "Initializing piece positions...\n";
    for (int z = 0; z < 8; z++) {
        for (int x = 0; x < 8; x++) {
            if (chessBoard.board[z][x] != PieceType::EMPTY) {
                MovingPiece piece;
                piece.current = {x * 100.0f + 50.0f, 10.0f, z * 100.0f + 50.0f};
                piece.target = piece.current;
                piece.isMoving = false;
                piece.moveSpeed = MOVE_SPEED;
                
                // Fix color determination - pieces in rows 0-1 are black, rows 6-7 are white
                bool isWhite = (z >= 6);  // Changed from (z < 4)
                piece.isWhite = isWhite;
                
                piecePositions.push_back(piece);
                std::cout << "Added " << (piece.isWhite ? "white" : "black") 
                         << " piece at: (" << x << ", " << z << ")\n";
            }
        }
    }
    std::cout << "Total pieces initialized: " << piecePositions.size() << "\n";
}

// Add this function to update piece positions
void updatePiecePositions() {
    for (auto& piece : piecePositions) {
        if (piece.isMoving) {
            std::cout << "Moving piece from (" << piece.current.x << ", " << piece.current.z 
                      << ") to (" << piece.target.x << ", " << piece.target.z << ")\n";
            
            // Update X position
            if (piece.current.x < piece.target.x) {
                piece.current.x = std::min(piece.current.x + MOVE_SPEED, piece.target.x);
            } else if (piece.current.x > piece.target.x) {
                piece.current.x = std::max(piece.current.x - MOVE_SPEED, piece.target.x);
            }

            // Update Z position
            if (piece.current.z < piece.target.z) {
                piece.current.z = std::min(piece.current.z + MOVE_SPEED, piece.target.z);
            } else if (piece.current.z > piece.target.z) {
                piece.current.z = std::max(piece.current.z - MOVE_SPEED, piece.target.z);
            }

            // Check if piece has reached target
            if (piece.current.x == piece.target.x && piece.current.z == piece.target.z) {
                piece.isMoving = false;
                std::cout << "Piece reached target position\n";
            }
        }
    }
}

int main() {
    chessBoard.printBoard();
    std::cout << "Initializing OpenGL..." << std::endl;
    GLFWwindow* window = initOpenGL();
    if (!window) return -1;

    // Initialize the chessboard and piece positions
    initializePiecePositions();
    std::cout << "Initialized " << piecePositions.size() << " pieces\n";

    // Camera setup
    float cameraPos[] = {0.0f, 0.0f, 5.0f}; 
    float target[] = {0.0f, 0.0f, 0.0f};
    float up[] = {0.0f, 1.0f, 0.0f};

    // Initialize view matrix
    setIdentityMatrix(viewMatrix); // Ensure viewMatrix is initialized
    setupViewMatrix(viewMatrix, cameraPos, target, up); // Properly set up the view matrix

    // Initialize projection matrix
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    float fov = 45.0f;
    float near = 0.1f;
    float far = 10000.0f; 
    perspectiveMatrix(projectionMatrix, fov, aspectRatio, near, far); 

    // Set mouse button callback
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    // Initialize pieces
    float pieceSize = 10.0f; 

    // White pieces
    int pawnxy = 50;
    for (int i = 0; i < 8; ++i) {
        pieces.push_back(Piece(pawnxy, 10.0f, 145.0f, pieceSize, false)); // White pawns
        pawnxy += 100;
    }
    int cnt = 45;
    int a = 102;
    pieces.push_back(Piece(cnt, 10.0f, 45.0f, pieceSize, false)); // White rook
    cnt += a;
    pieces.push_back(Piece(cnt, 10.0f, 45.0f, pieceSize, false)); // White knight
    cnt += a;
    pieces.push_back(Piece(cnt, -20.0f, 45.0f, pieceSize, false)); // White bishop
    cnt += a;
    pieces.push_back(Piece(cnt, -20.0f, 45.0f, pieceSize, false)); // White king
    cnt += a;
    pieces.push_back(Piece(cnt, -20.0f, 45.0f, pieceSize, false)); // White queen
    cnt += a;
    pieces.push_back(Piece(cnt, -20.0f, 45.0f, pieceSize, false)); // White bishop
    cnt += a;
    pieces.push_back(Piece(cnt, 10.0f, 45.0f, pieceSize, false)); // White knight
    cnt += a;
    pieces.push_back(Piece(cnt, 10.0f, 45.0f, pieceSize, false)); // White rook

    // Black pieces
    pawnxy = 50;
    for (int i = 0; i < 8; ++i) {
        pieces.push_back(Piece(pawnxy, 10.0f, 675.0f, pieceSize, false)); // Black pawns
        pawnxy += 100;
    }
    cnt = 45;
    pieces.push_back(Piece(cnt, 10.0f, 745.0f, pieceSize, false)); // Black rook
    cnt += a;
    pieces.push_back(Piece(cnt, 10.0f, 745.0f, pieceSize, false)); // Black knight
    cnt += a;
    pieces.push_back(Piece(cnt, -20.0f, 745.0f, pieceSize, false)); // Black bishop
    cnt += a;
    pieces.push_back(Piece(cnt, -20.0f, 745.0f, pieceSize, false)); // Black king
    cnt += a;
    pieces.push_back(Piece(cnt, -20.0f, 745.0f, pieceSize, false)); // Black queen
    cnt += a;
    pieces.push_back(Piece(cnt, -20.0f, 745.0f, pieceSize, false)); // Black bishop
    cnt += a;
    pieces.push_back(Piece(cnt, 10.0f, 745.0f, pieceSize, false)); // Black knight
    cnt += a;
    pieces.push_back(Piece(cnt, 10.0f, 745.0f, pieceSize, false)); // Black rook

    GLuint shaderProgram = compileShaders();

    std::vector<float> boardVertices;
    std::vector<unsigned int> boardIndices;
    setupChessboard(boardVertices, boardIndices);

    GLuint boardVAO = setupVAO(boardVertices, boardIndices);

    // Load models for each piece type
    std::vector<float> pawnVertices, rookVertices, knightVertices, bishopVertices, queenVertices, kingVertices;
    std::vector<unsigned int> pawnIndices, rookIndices, knightIndices, bishopIndices, queenIndices, kingIndices;

    loadModel("Pawn.obj", pawnVertices, pawnIndices);
    loadModel("Rook.obj", rookVertices, rookIndices);
    loadModel("Knight.obj", knightVertices, knightIndices);
    loadModel("Bishop.obj", bishopVertices, bishopIndices);
    loadModel("Queen.obj", queenVertices, queenIndices);
    loadModel("King.obj", kingVertices, kingIndices);

    GLuint pawnVAO = setupVAO(pawnVertices, pawnIndices);
    GLuint rookVAO = setupVAO(rookVertices, rookIndices);
    GLuint knightVAO = setupVAO(knightVertices, knightIndices);
    GLuint bishopVAO = setupVAO(bishopVertices, bishopIndices);
    GLuint queenVAO = setupVAO(queenVertices, queenIndices);
    GLuint kingVAO = setupVAO(kingVertices, kingIndices);

    // Load models for black pieces and set their color
    std::vector<float> blackPawnVertices, blackRookVertices, blackKnightVertices, blackBishopVertices, blackQueenVertices, blackKingVertices;
    std::vector<unsigned int> blackPawnIndices, blackRookIndices, blackKnightIndices, blackBishopIndices, blackQueenIndices, blackKingIndices;

    loadModel("Pawn.obj", blackPawnVertices, blackPawnIndices);
    setBlackPieceColor(blackPawnVertices);
    GLuint blackPawnVAO = setupVAO(blackPawnVertices, blackPawnIndices);

    loadModel("Rook.obj", blackRookVertices, blackRookIndices);
    setBlackPieceColor(blackRookVertices);
    GLuint blackRookVAO = setupVAO(blackRookVertices, blackRookIndices);

    loadModel("Knight.obj", blackKnightVertices, blackKnightIndices);
    setBlackPieceColor(blackKnightVertices);
    GLuint blackKnightVAO = setupVAO(blackKnightVertices, blackKnightIndices);

    loadModel("Bishop.obj", blackBishopVertices, blackBishopIndices);
    setBlackPieceColor(blackBishopVertices);
    GLuint blackBishopVAO = setupVAO(blackBishopVertices, blackBishopIndices);

    loadModel("Queen.obj", blackQueenVertices, blackQueenIndices);
    setBlackPieceColor(blackQueenVertices);
    GLuint blackQueenVAO = setupVAO(blackQueenVertices, blackQueenIndices);

    loadModel("King.obj", blackKingVertices, blackKingIndices);
    setBlackPieceColor(blackKingVertices);
    GLuint blackKingVAO = setupVAO(blackKingVertices, blackKingIndices);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    int test = 0;

    std::cout << "Starting render loop..." << std::endl;
    //while (!glfwWindowShouldClose(window)) {
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);

        // Update piece positions
        updatePiecePositions();

        float view[16], projection[16];
        setIdentityMatrix(view);
        translateMatrix(view, -370.0f, -160.5f, -2000.0f); // -19.0f
        rotateMatrixX(view, -40.0f);

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
        float fov = 45.0f;
        float near = 0.1f;
        float far = 10000.0f; 
        perspectiveMatrix(projection, fov, aspectRatio, near, far);

        //printMatrix(view, "View Matrix");
        //printMatrix(projection, "Projection Matrix");

        GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
        GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view);
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection);

        float boardModel[16];
        setIdentityMatrix(boardModel);
        //printMatrix(boardModel, "Model Matrix");
        renderPiece(shaderProgram, boardVAO, boardModel, boardIndices.size());

        float pieceScale = 0.4f; 
        float rotationAngleX = 0.0f;
        float rotationAngleY = 270.0f; 
        float rotationAngleZ = 90.0f; 

        // Render white pieces
        pawnxy = 50;
        int cnt = 45;
        int a = 102;
      

        // Add debug visualization
        drawDebugGrid(shaderProgram);

        // Render pieces at their current positions
        for (int i = 0; i < piecePositions.size(); i++) {
            const auto& piece = piecePositions[i];
            PieceType pieceType = chessBoard.board[static_cast<int>(piece.current.z/100.0f)][static_cast<int>(piece.current.x/100.0f)];
            bool isWhite = (piece.current.z < 400.0f);
            
            // Select the appropriate VAO based on piece type and color
            GLuint vao;
            size_t indicesCount;
            
            switch(pieceType) {
                case PieceType::KING_WHITE:
                    vao = kingVAO;
                    indicesCount = kingIndices.size();
                    break;
                case PieceType::KING_BLACK:
                    vao = blackKingVAO;
                    indicesCount = blackKingIndices.size();
                    break;
                case PieceType::QUEEN_WHITE:
                    vao = queenVAO;
                    indicesCount = queenIndices.size();
                    break;
                case PieceType::QUEEN_BLACK:
                    vao = blackQueenVAO;
                    indicesCount = blackQueenIndices.size();
                    break;
                case PieceType::ROOK_WHITE:
                    vao = rookVAO;
                    indicesCount = rookIndices.size();
                    break;
                case PieceType::ROOK_BLACK:
                    vao = blackRookVAO;
                    indicesCount = blackRookIndices.size();
                    break;
                case PieceType::KNIGHT_WHITE:
                    vao = knightVAO;
                    indicesCount = knightIndices.size();
                    break;
                case PieceType::KNIGHT_BLACK:
                    vao = blackKnightVAO;
                    indicesCount = blackKnightIndices.size();
                    break;
                case PieceType::BISHOP_WHITE:
                    vao = bishopVAO;
                    indicesCount = bishopIndices.size();
                    break;
                case PieceType::BISHOP_BLACK:
                    vao = blackBishopVAO;
                    indicesCount = blackBishopIndices.size();
                    break;
                case PieceType::PAWN_WHITE:
                    vao = pawnVAO;
                    indicesCount = pawnIndices.size();
                    break;
                case PieceType::PAWN_BLACK:
                    vao = blackPawnVAO;
                    indicesCount = blackPawnIndices.size();
                    break;
                default:
                    continue;
            }
            
            // Check if this piece is selected
            bool isSelected = (selectedPiece.isSelected && 
                              selectedPiece.x == static_cast<int>(piece.current.x/100.0f) && 
                              selectedPiece.z == static_cast<int>(piece.current.z/100.0f));
            
            // Render the piece with appropriate parameters
            renderPieceAt(shaderProgram, vao, 
                         piece.current.x, piece.current.y, piece.current.z,
                         indicesCount, pieceScale, 
                         rotationAngleX, rotationAngleY, rotationAngleZ, 
                         isSelected);
        }

        std::vector<float> signatureQuadVertices = {
            // positions          // colors (red)
            -200.0f,  150.0f, 0.0f,   1.0f, 0.0f, 0.0f,  // top left
             200.0f,  150.0f, 0.0f,   1.0f, 0.0f, 0.0f,  // top right
             200.0f, -150.0f, 0.0f,   1.0f, 0.0f, 0.0f,  // bottom right
            -200.0f, -150.0f, 0.0f,   1.0f, 0.0f, 0.0f   // bottom left
        };

        std::vector<unsigned int> signatureQuadIndices = {
            0, 1, 2,   // first triangle
            2, 3, 0    // second triangle
        };

        GLuint signatureVAO = setupVAO(signatureQuadVertices, signatureQuadIndices);

        // Render signature quad
        float signatureModel[16];
        setIdentityMatrix(signatureModel);
        translateMatrix(signatureModel, 370.0f, 200.0f, -500.0f);
        renderPiece(shaderProgram, signatureVAO, signatureModel, signatureQuadIndices.size());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    std::cout << "Cleaning up..." << std::endl;
    glDeleteVertexArrays(1, &boardVAO);
    glDeleteVertexArrays(1, &pawnVAO);
    glDeleteVertexArrays(1, &rookVAO);
    glDeleteVertexArrays(1, &knightVAO);
    glDeleteVertexArrays(1, &bishopVAO);
    glDeleteVertexArrays(1, &queenVAO);
    glDeleteVertexArrays(1, &kingVAO);
    glDeleteVertexArrays(1, &blackPawnVAO);
    glfwDestroyWindow(window);
    glfwTerminate();
    std::cout << "Done." << std::endl;
    return 0;
}