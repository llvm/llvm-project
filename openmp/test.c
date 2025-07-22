#include <omp.h>
#include <stdio.h>

int main() {
    int x = 0;
    #pragma omp parallel for
    for (int i = 0; i < 10; i++) {
        x += i;
    }
    printf("x = %d\n", x);
    return 0;
}
// Updated to trigger bot

