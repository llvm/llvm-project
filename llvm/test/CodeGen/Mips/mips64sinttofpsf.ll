; RUN: llc -mtriple=mips64 -mcpu=mips64r2 -mattr=+soft-float -O0 < %s | FileCheck %s


define double @foo() #0 {
entry:
  %x = alloca i32, align 4
  store volatile i32 -32, ptr %x, align 4
  %0 = load volatile i32, ptr %x, align 4
  %conv = sitofp i32 %0 to double
  ret double %conv

; CHECK-NOT:        dsll
; CHECK-NOT:        dsrl

}
