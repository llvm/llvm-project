// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

unsigned long do_csum(const unsigned char *buff, int len, unsigned long result) {
  if (2 & (unsigned long) buff) result += 1;
  return result;
}
