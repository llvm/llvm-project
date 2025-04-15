// RUN: not llvm-mc --triple thumbv6m %s 2>&1 | FileCheck %s --check-prefixes=CHECK,THUMB1
// RUN: not llvm-mc --triple thumbv7m %s 2>&1 | FileCheck %s --check-prefixes=CHECK,THUMB2

// This test checks reporting of errors of the form "you should have
// used :lower16: in this immediate field", during initial reading of
// the source file.
//
// For errors that are reported at object-file output time, see
// lower-upper-errors-2.s.

// CHECK: :[[@LINE+1]]:10: error: Immediate expression for Thumb movs requires :lower0_7:, :lower8_15:, :upper0_7: or :upper8_15:
movs r0, #foo

// CHECK: :[[@LINE+1]]:10: error: Immediate expression for Thumb adds requires :lower0_7:, :lower8_15:, :upper0_7: or :upper8_15:
adds r0, #foo

// THUMB2: :[[@LINE+1]]:10: error: immediate expression for mov requires :lower16: or :upper16
movw r0, #foo

// THUMB2: :[[@LINE+1]]:10: error: immediate expression for mov requires :lower16: or :upper16
movt r0, #foo

// With a Thumb2 wide add instruction available, this case isn't an error
// while reading the source file. It only causes a problem when an object
// file is output, and it turns out there's no suitable relocation to ask
// for the value of an external symbol to be turned into a Thumb shifted
// immediate field. And in this case the other errors in this source file
// cause assembly to terminate before reaching the object-file output stage
// (even if a test run had had -filetype=obj).

// THUMB1: :[[@LINE+2]]:14: error: Immediate expression for Thumb adds requires :lower0_7:, :lower8_15:, :upper0_7: or :upper8_15:
// THUMB2-NOT: :[[@LINE+1]]:{{[0-9]+}}: error:
adds r0, r0, #foo

// Similarly for this version, which _must_ be the wide encoding due
// to the use of a high register and the lack of flag-setting.

// THUMB1: :[[@LINE+2]]:1: error: invalid instruction
// THUMB2-NOT: :[[@LINE+1]]:{{[0-9]+}}: error:
add r9, r0, #foo

// CHECK: :[[@LINE+1]]:10: error: Immediate expression for Thumb movs requires :lower0_7:, :lower8_15:, :upper0_7: or :upper8_15:
movs r0, :lower16:#foo

// This is invalid in either architecture: in Thumb1 it can't use a
// high register, and in Thumb2 it can't use :upper8_15:. But the
// Thumb2 case won't cause an error until output.

// THUMB1: :[[@LINE+2]]:1: error: invalid instruction
// THUMB2-NOT: :[[@LINE+1]]:{{[0-9]+}}: error:
movs r11, :upper8_15:#foo
