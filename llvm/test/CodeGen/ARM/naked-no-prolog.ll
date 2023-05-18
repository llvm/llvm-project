; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a7 -verify-machineinstrs %s -o - | FileCheck %s

%struct.S = type { [65 x i8] }

define void @naked_no_prologue(ptr byval(%struct.S) align 4 %0) naked noinline nounwind optnone {
; CHECK-NOT: stm
; CHECK-NOT: str

entry:
  ret void
  unreachable
}

