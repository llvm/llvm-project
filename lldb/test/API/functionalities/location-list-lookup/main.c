#include <stdio.h>
#include <stdlib.h>

// The goal with this test is:
//  1. Have main() followed by foo()
//  2. Have the no-return call to abort() in main be the last instruction
//  3. Have the next instruction be the start of foo()
//  4. The debug info for argv uses a location list.
//     clang at -O1 on x86_64 or arm64 has debuginfo like
//          DW_AT_location	(0x00000049:
//              [0x0000000100003f15, 0x0000000100003f25): DW_OP_reg4 RSI
//              [0x0000000100003f25, 0x0000000100003f5b): DW_OP_reg15 R15)

void foo(int);
int main(int argc, char **argv) {
  char *file = argv[0];
  char f0 = file[0];
  printf("%c\n", f0);
  foo(f0);
  printf("%s %d\n", argv[0], argc);
  abort(); /// argv is still be accessible here
}
void foo(int in) { printf("%d\n", in); }
