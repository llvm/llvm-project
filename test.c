#include <math.h>
#include "./myheader.c"
#include <stdio.h>

void imported_function(float* data, int size);

/* Function 1:independent iterations (non parallelizable) */
void Function1(float* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] > 0) {
            data[i] = sqrt(data[i]);
        }
    }
}

/* Function 2: Independent loop with sin and cos (parallelizable) */
void Function2(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = sin(data[i]) + cos(data[i]);
    }
}

/* Function 3:independent loop with sin(parallelizable) */
void Function3(float* data, int size) {
    int src=0;
    for (int i = 0; i < size; i++) {
        data[i] = data[i]*2;
    }
}

/* Function 4: Loop with data dependency (not parallelizable) */
void Functino4(float* data, int size) {
    for (int i = 1; i < size; i++) {
        data[i] = data[i - 1] + data[i];
    }
}

/* Function 5:independent loop for 2D(parallelizable) */
void Function5(float** data, int size) {
    for (int i = 1; i < size; i++) {
        data[i][i] = data[i][i]*2;
    }
}

//Function 6: Loop with data dependency (not parallelizable)
void Function6(float* data, int size) {
    int src;
    for (int i = 0; i < size; i++) {
        src = src + data[i];
    }
}

//Function 7: Empty Loop
void Function7(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = data[i*2] * 2.0f;
    }
}

// Function 8: Nested loops (parallelizable)
void Function8(float* data, int size) {
    for (int i = 0; i < size; i++) {
    }
}

// Function 9:trial in final selection round
int arr[10] = {0,1,2,3,4,5,6,7,8,9};
void final(float* data, int size) {
    for (int i = 0; i < 5; i++) {
        data[i] = arr[i]*arr[i];
    }
}

void trail(float* data,int size){
    int src =0;
    for(int i=0;i<size;i++){
        src += data[i];
    }
}
