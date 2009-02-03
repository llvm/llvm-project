/* RUN: %llvmgcc -xc %s -S -o - | grep -v alloca | not grep bitcast
 */

void test(int* array, long long N) {
    array[N] = N[array] = 33;
}

