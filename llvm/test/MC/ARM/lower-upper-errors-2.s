// RUN: not llvm-mc --triple thumbv7m -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s

// This test checks reporting of errors of the form "you should have
// used :lower16: in this immediate field", when the errors are
// discovered at the object-file output stage by checking the set of
// available relocations.
//
// For errors that are reported earlier, when initially reading the
// instructions, see lower-upper-errors.s.

// CHECK: [[@LINE+1]]:1: error: unsupported relocation
adds r0, r0, #foo

// CHECK: [[@LINE+1]]:1: error: unsupported relocation
add r9, r0, #foo

// CHECK: [[@LINE+1]]:1: error: expected relocatable expression
movs r11, :upper8_15:#foo
