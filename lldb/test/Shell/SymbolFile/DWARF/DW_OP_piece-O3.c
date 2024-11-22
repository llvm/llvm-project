// Check that optimized with -O3 values that have a file address can be read
// DWARF info:
// 0x00000023:   DW_TAG_variable
//                 DW_AT_name      ("array")
//                 DW_AT_type      (0x00000032 "char[5]")
//                 DW_AT_location  (DW_OP_piece 0x2, DW_OP_addrx 0x0, DW_OP_piece 0x1)

// RUN: %clang_host -O3 -gdwarf %s -o %t
// RUN: %lldb %t \
// RUN:   -o "b done" \
// RUN:   -o "r" \
// RUN:   -o "p/x array[2]" \
// RUN:   -b | FileCheck %s
//
// CHECK: (lldb) p/x array[2]
// CHECK: (char) 0x03

static char array[5] = {0, 1, 2, 3, 4};

int done() __attribute__((noinline));
int done() { return array[2]; };

int main(void) {
  ++array[2];
  return done();
}
