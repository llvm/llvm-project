#if defined(__x86_64__) || defined(__i386__)
#define BRANCH_FWD "jmp 1f\n"
#elif defined(__aarch64__) || defined(__arm__) || defined(__powerpc64__) ||    \
    defined(__powerpc__) || defined(__s390x__) || defined(__loongarch__)
#define BRANCH_FWD "b 1f\n"
#elif defined(__riscv)
#define BRANCH_FWD "j 1f\n"
#else
#error Do not know unconditional branch mnemonic for this architecture.
#endif

volatile int always_false;

int foo() {
  asm volatile(BRANCH_FWD
               // This is dead code that we are allowed to corrupt.
               // There are 4 so that even if nop is 1 byte, we have 4 bytes
               // we can overwrite safely.
               "nop\n"
               "nop\n"
               "nop\n"
               "nop\n"
               // No live code should be between the dead code above
               // and the breakpoint location below.
               "1:\n"
               ".globl place_break_here\n"
               "place_break_here:\n"
               // The test will repeatedly add and remove a breakpoint here.
               "nop");

  return 0;
}

int main() {
  volatile int sum = 0;
  // Pick a number of loops >= the number of write patterns being tested.
  for (unsigned i = 0; i < 10; ++i)
    sum += foo(); // break here

  return 0;
}
