; RUN: llc -mtriple=riscv32 -mattr=+reserve-x3 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X3
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x3 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X3
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x4 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X4
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x4 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X4
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x5 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X5
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x5 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X5
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x6 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X6
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x6 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X6
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x7 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X7
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x7 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X7
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x8 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X8
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x8 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X8
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x9 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X9
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x9 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X9
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x10 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X10
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x10 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X10
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x11 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X11
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x11 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X11
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x12 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X12
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x12 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X12
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x13 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X13
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x13 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X13
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x14 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X14
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x14 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X14
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x15 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X15
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x15 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X15
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x16 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X16
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x16 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X16
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x17 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X17
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x17 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X17
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x18 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X18
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x18 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X18
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x19 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X19
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x19 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X19
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x20 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X20
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x20 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X20
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x21 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X21
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x21 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X21
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x22 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X22
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x22 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X22
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x23 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X23
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x23 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X23
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x24 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X24
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x24 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X24
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x25 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X25
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x25 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X25
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x26 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X26
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x26 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X26
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x27 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X27
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x27 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X27
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x28 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X28
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x28 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X28
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x29 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X29
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x29 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X29
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x30 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X30
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x30 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X30
; RUN: llc -mtriple=riscv32 -mattr=+reserve-x31 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X31
; RUN: llc -mtriple=riscv64 -mattr=+reserve-x31 -verify-machineinstrs < %s | FileCheck %s -check-prefix=X31

; This program is free to use all registers, but needs a stack pointer for
; spill values, so do not test for reserving the stack pointer.

; Used to exhaust all registers
@var = global [32 x i64] zeroinitializer

define void @foo() {
  %1 = load volatile [32 x i64], ptr @var
  store volatile [32 x i64] %1, ptr @var

; X3-NOT: lw gp,
; X3-NOT: ld gp,
; X4-NOT: lw tp,
; X4-NOT: ld tp,
; X5-NOT: lw t0,
; X5-NOT: ld t0,
; X6-NOT: lw t1,
; X6-NOT: ld t1,
; X7-NOT: lw t2,
; X7-NOT: ld t2,
; X8-NOT: lw s0,
; X8-NOT: ld s0,
; X9-NOT: lw s1,
; X9-NOT: ld s1,
; X10-NOT: lw a0,
; X10-NOT: ld a0,
; X11-NOT: lw a1,
; X11-NOT: ld a1,
; X12-NOT: lw a2,
; X12-NOT: ld a2,
; X13-NOT: lw a3,
; X13-NOT: ld a3,
; X14-NOT: lw a4,
; X14-NOT: ld a4,
; X15-NOT: lw a5,
; X15-NOT: ld a5,
; X16-NOT: lw a6,
; X16-NOT: ld a6,
; X17-NOT: lw a7,
; X17-NOT: ld a7,
; X18-NOT: lw s2,
; X18-NOT: ld s2,
; X19-NOT: lw s3,
; X19-NOT: ld s3,
; X20-NOT: lw s4,
; X20-NOT: ld s4,
; X21-NOT: lw s5,
; X21-NOT: ld s5,
; X22-NOT: lw s6,
; X22-NOT: ld s6,
; X23-NOT: lw s7,
; X23-NOT: ld s7,
; X24-NOT: lw s8,
; X24-NOT: ld s8,
; X25-NOT: lw s9,
; X25-NOT: ld s9,
; X26-NOT: lw s10,
; X26-NOT: ld s10,
; X27-NOT: lw s11,
; X27-NOT: ld s11,
; X28-NOT: lw t3,
; X28-NOT: ld t3,
; X29-NOT: lw t4,
; X29-NOT: ld t4,
; X30-NOT: lw t5,
; X30-NOT: ld t5,
; X31-NOT: lw t6,
; X31-NOT: ld t6,

  ret void
}
