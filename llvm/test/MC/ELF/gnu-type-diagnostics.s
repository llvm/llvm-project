// RUN: not llvm-mc -triple i686-elf -filetype asm -o /dev/null %s 2>&1 | FileCheck %s

	.type TYPE FUNC
// CHECK: error: unsupported attribute
// CHECK: .type TYPE FUNC
// CHECK:            ^

	.type type stt_func
// CHECK: error: unsupported attribute
// CHECK: .type type stt_func
// CHECK:            ^

	.type symbol 32
// CHECK: error: expected STT_<TYPE_IN_UPPER_CASE>, '#<type>', '@<type>', '%<type>' or "<type>"
// CHECK: .type symbol 32
// CHECK:              ^


