// RUN: %llvmgcc -S %s -o /dev/null -Wall -Werror
void bork() {
  char * volatile p;
  volatile int cc;
  p += cc;
}
