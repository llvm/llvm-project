volatile int always_false;

int foo() {
  // This nop is deliberately dead code so that we have an instruction
  // immediately prior to the breakpoint that we are allowed to corrupt.
  if (always_false) {
    // This dead code must be at least 4 bytes even on an architecture where
    // nop is 1 byte.
    asm volatile("nop\n"
                 "nop\n"
                 "nop\n"
                 "nop\n");
  }
  // We are assuming that there are no instructions placed between the dead code
  // above and the assembly below. This is not guaranteed but it's safer than
  // assuming that this function will have no prologue instructions or breaking
  // in the very first instruction of foo and hoping whatever comes before is
  // not important.
  asm volatile(".globl place_break_here\n"
#ifdef _WIN32
               ".def place_break_here; .scl 2; .type 32; .endef;\n"
#endif
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
