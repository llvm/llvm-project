#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

void handler(int sig) {
  // The kernel only changes a few registers so set them all to something other
  // than the values in sigill() so that we can't fall back to real registers
  // and still pass the test.
#define SETREG(N) "mov x" N ", #" N "+1\n\t"
  asm volatile(
      /* clang-format off */
      /* x0 is used for a parameter */
                   SETREG("1")  SETREG("2")  SETREG("3")
      SETREG("4")  SETREG("5")  SETREG("6")  SETREG("7")
      SETREG("8")  SETREG("9")  SETREG("10") SETREG("11")
      SETREG("12") SETREG("13") SETREG("14") SETREG("15")
      SETREG("16") SETREG("17") SETREG("18") SETREG("19")
      SETREG("20") SETREG("21") SETREG("22") SETREG("23")
      SETREG("24") SETREG("25") SETREG("26") SETREG("27")
      SETREG("28") // fp/x29 needed for unwiding
      SETREG("30") // 31 is xzr/sp
      /* clang-format on */
      ::
          : /* skipped x0 */ "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
            "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18",
            "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27",
            "x28",
            /* skipped fp/x29 */ "x30");
  printf("Set a breakpoint here.\n");
  exit(0);
}

static void sigill() {
  // Set all general registers to known values to check
  // that the signal unwind plan sets their locations correctly.
#define SETREG(N) "mov x" N ", #" N "\n\t"
  asm volatile(
      /* clang-format off */
      SETREG("0")  SETREG("1")  SETREG("2")  SETREG("3")
      SETREG("4")  SETREG("5")  SETREG("6")  SETREG("7")
      SETREG("8")  SETREG("9")  SETREG("10") SETREG("11")
      SETREG("12") SETREG("13") SETREG("14") SETREG("15")
      SETREG("16") SETREG("17") SETREG("18") SETREG("19")
      SETREG("20") SETREG("21") SETREG("22") SETREG("23")
      SETREG("24") SETREG("25") SETREG("26") SETREG("27")
      SETREG("28") SETREG("29") SETREG("30") /* 31 is xzr/sp */
      /* clang-format on */
      ".inst   0x00000000\n\t" // udf #0 (old binutils don't support udf)
      ::
          : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
            "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19",
            "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28",
            "x29", "x30");
}

int main() {
  if (signal(SIGILL, handler) == SIG_ERR) {
    perror("signal");
    return 1;
  }

  sigill();
  return 2;
}
