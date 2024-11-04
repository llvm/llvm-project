// RUN: not llvm-mc --triple thumbv6m -show-encoding %s 2>&1 | FileCheck %s
// RUN: not llvm-mc --triple thumbv7m -show-encoding %s 2>&1 | FileCheck %s --check-prefixes=CHECK,THUMB2

// Check reporting of errors of the form "you should have used
// :lower16: in this immediate field".

// CHECK: :[[@LINE+1]]:10: error: Immediate expression for Thumb movs requires :lower0_7:, :lower8_15:, :upper0_7: or :upper8_15:
movs r0, #foo

// CHECK: :[[@LINE+1]]:10: error: Immediate expression for Thumb adds requires :lower0_7:, :lower8_15:, :upper0_7: or :upper8_15:
adds r0, #foo

// CHECK: :[[@LINE+1]]:14: error: Immediate expression for Thumb adds requires :lower0_7:, :lower8_15:, :upper0_7: or :upper8_15:
adds r0, r0, #foo

// THUMB2: :[[@LINE+1]]:10: error: immediate expression for mov requires :lower16: or :upper16
movw r0, #foo

// THUMB2: :[[@LINE+1]]:10: error: immediate expression for mov requires :lower16: or :upper16
movt r0, #foo
