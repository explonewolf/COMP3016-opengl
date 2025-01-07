#include <SDL.h>
#include <iostream>
// https://greenchess.net/info.php?item=downloads
//https://www.chess.com/forum/view/general/how-to-get-the-pieces-more-as-png
using namespace std;
int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* win = SDL_CreateWindow("Triangle in SDL", 100, 100, 800, 600, SDL_WINDOW_SHOWN);
    if (win == NULL) {
        printf("SDL_CreateWindow Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (renderer == NULL) {
        SDL_DestroyWindow(win);
        printf("SDL_CreateRenderer Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    // Main loop
    int running = 1;
    SDL_Event event;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = 0;
            }
        }

        // Clear screen
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // Black
        SDL_RenderClear(renderer);

        // Draw triangle
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White
        SDL_RenderDrawLine(renderer, 400, 100, 600, 400); // Point 1 -> Point 2
        SDL_RenderDrawLine(renderer, 600, 400, 200, 400); // Point 2 -> Point 3
        SDL_RenderDrawLine(renderer, 200, 400, 400, 100); // Point 3 -> Point 1

        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
