// compute.c
#include <stdio.h>

void imported_function(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = data[i] * data[i];
    }
}
