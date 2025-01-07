# OpenGL Chess Game
3D models - https://github.com/stevenalbert/3d-chess-opengl/tree/master/model
## Dependencies Used
- OpenGL: Core graphics API for rendering
- GLFW: Window management and OpenGL context creation
- SDL: Used for texture loading
- GLM: Mathematics library for 3D transformations

## Game Programming Patterns
1. **State Pattern**
   - Used for managing piece selection and movement states
   - Implemented in the `MovingPiece` structure to track piece positions and movement

2. **Component Pattern**
   - Separation of rendering and game logic
   - Modular approach to handling shaders and VAO setup

## Game Mechanics
1. **Piece Movement**
   - Implemented in `updatePiecePositions()` function
   - Uses time-based movement for consistent speed

2. **Selection System**
   - Mouse-based piece selection
   - Grid-based movement validation
   - Implemented in `handleSquareSelection()` function

## Software Engineering Considerations
### Performance vs. Good Practice Trade-offs
1. **Matrix Operations**
   - Custom matrix operations for better control
   - Could have used GLM for better maintainability

2. **Shader Management**
   - Inline shader code for simplicity
   - Could be improved by loading from external files

## Exception Handling and Testing
1. **Shader Compilation**
   - Error checking for shader compilation and linking
   - Logs compilation errors to console

2. **OpenGL State Validation**
   - Checks for successful context creation
   - Validates texture loading

## Implementation Details
1. **Rendering Pipeline**
   - Vertex and Fragment shaders for 3D rendering
   - Custom matrix transformations
   - Texture mapping for signature display

2. **User Interface**
   - Mouse interaction for piece selection
   - Visual feedback for selected pieces

## Evaluation
### Achievements
- Created a functional 3D chessboard
- Implemented texture rendering for signature display

### Future Improvements
1. Better separation of concerns in code structure
2. Implementation of actual chess rules
3. Enhanced visual effects and animations
4. better rending of objects

### Personal Learning
- Gained practical experience with OpenGL
- Improved understanding of 3D graphics programming
- Learned about game state management
