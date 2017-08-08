#include <stdio.h>

#include <cilk/cilk.h>

#define N 100000000

__attribute__((always_inline))
int f(int x) {
    return x * x;
}

__attribute__((always_inline))
int g(int x) {
    return x + 3;
}

int r1[N];
int r2[N];

int main(void)
{
    int sum = 0;

    cilk_for (int i=0; i<N; i++) {
        r1[i] = f(i) * g(i);
    }

    cilk_for (int i=0; i<N; i++) {
        r2[i] = f(i) / g(i);
    }

    printf("%d %d\n", r1[N / 2], r2[N / 2]);

    return 0;
}
