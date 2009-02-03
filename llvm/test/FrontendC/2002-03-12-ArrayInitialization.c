// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

/* GCC would generate bad code if not enough initializers are 
   specified for an array.
 */

int a[10] = { 0, 2};

char str[10] = "x";

void *Arr[5] = { 0, 0 };

float F[12] = { 1.23f, 34.7f };

struct Test { int X; double Y; };

struct Test Array[10] = { { 2, 12.0 }, { 3, 24.0 } };

int B[4][4] = { { 1, 2, 3, 4}, { 5, 6, 7 }, { 8, 9 } };
