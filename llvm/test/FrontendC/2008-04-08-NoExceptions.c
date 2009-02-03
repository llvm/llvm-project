// RUN: %llvmgcc -S -o - %s | grep nounwind | count 2
// RUN: %llvmgcc -S -o - %s | not grep {declare.*nounwind}

void f(void);
void g(void) {
  f();
}
