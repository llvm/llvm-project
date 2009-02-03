// Test that basic generic vector support works
// RUN: %llvmgcc %s -S -o -

typedef int v4si __attribute__ ((__vector_size__ (16)));
void test(v4si *P, v4si *Q, float X) {
  *P = (v4si){ X, X, X, X } * *Q;
}

v4si G = (v4si){ 0.1, 1.2, 4.2, 17.2 };

