// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

int foo(int len, char arr[][len], int X) {
  return arr[X][0];
}
