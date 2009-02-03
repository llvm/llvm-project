// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

int tcount;
void test(char *, const char*, int);
void foo() {
  char Buf[10];
  test(Buf, "n%%%d", tcount++);
}
