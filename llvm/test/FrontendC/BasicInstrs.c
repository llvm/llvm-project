// This file can be used to see what a native C compiler is generating for a
// variety of interesting operations.
//
// RUN: %llvmgcc -S %s -o - | llvm-as | llc

unsigned int udiv(unsigned int X, unsigned int Y) {
  return X/Y;
}
int sdiv(int X, int Y) {
  return X/Y;
}
unsigned int urem(unsigned int X, unsigned int Y) {
  return X%Y;
}
int srem(int X, int Y) {
  return X%Y;
}

_Bool setlt(int X, int Y) {
  return X < Y;
}

_Bool setgt(int X, int Y) {
  return X > Y;
}

