#include <stdlib.h>
unsigned int test1(unsigned int x, unsigned int y) {
    if ((x & y) == y)
        return x - y;
    abort();
}
unsigned int test2(unsigned int x, unsigned int y) {
    if ((x & y) == y)
        return x ^ y;
    abort();
}
unsigned int test3(unsigned int x, unsigned int y) {
    if ((x & y) == y)
        return x & ~y;
    abort();
}
