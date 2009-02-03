// RUN: %llvmgcc -S %s -o -  -fnested-functions
// PR1274

void Bork() {
  void Fork(const int *src, int size) {
    int i = 1;
    int x;

    while (i < size)
      x = src[i];
  }
}

void foo(void *a){
  inline void foo_bar() {
    a += 1;
  }
}
