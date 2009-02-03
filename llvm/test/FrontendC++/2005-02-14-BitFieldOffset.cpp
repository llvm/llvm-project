// RUN: %llvmgxx %s -S -o - | not grep {i32 6}

struct QVectorTypedData {
    int size;
    unsigned int sharable : 1;
    unsigned short array[1];
};

void foo(QVectorTypedData *X) {
  X->array[0] = 123;
}
