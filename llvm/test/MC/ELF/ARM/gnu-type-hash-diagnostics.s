@ RUN: not llvm-mc -triple arm-elf -filetype asm -o /dev/null %s 2>&1 | FileCheck %s

	.syntax unified

	.type TYPE #32
// CHECK: error: expected symbol type in directive
// CHECK: .type TYPE #32
// CHECK:             ^

  // For ARM, the comment character is '@', so we don't list '@<type>' as a
  // valid option.
  .section "foo", "a", @progbits
// CHECK: error: expected '%<type>' or "<type>"
