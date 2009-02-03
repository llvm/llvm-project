/* RUN: ignore */
#include <stdio.h>

/* Make this invalid C++ */
typedef struct {
    int i;
    char c;
} a;

static a b = { .i = 65, .c = 'r'};

void test() {
    b.i = 9;
    fflush(stdout);
    printf("el");
}

