// Function 1: Independent loop iterations (parallelizable)
void independent_loop(int* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = data[i] * 2;
    }
}
// Function 2: Loop with data dependency (not parallelizable)
void dependent_loop(int* data, int size) {
    for (int i = 1; i < size; i++) {
        data[i] = data[i - 1] + 1; 
    }
}
// Function 3: Nested loops
void nested_loop(int* data, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < 10; j++) {
            data[i] += j;
        }
    }
}
// Function 4: Independent loop with array access (parallelizable)
void sqr_function(int* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = data[i] * data[i];
    }
}
// Function 5: Loop with conditional dependency (not parallelizable)
void conditional_loop(int* data, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        if (data[i] > 0) {
            sum += data[i];
        }
        data[i] = sum;
    }
}


//Function 7:
int arr[10] = {0,1,2,3,4,5,6,7,8,9};
void final(int* data, int size) {
    for (int i = 0; i < 5; i++) {
        data[i] = arr[data[i]];
    }
}