#include <omp.h>
#include <stdio.h>

int main() {
    int sum = 0;
    #pragma omp parallel for
    for (int i = 0; i < 100; i++) {
        sum += i;
    }
    printf("Sum is %d\n", sum);
    return 0;
}
