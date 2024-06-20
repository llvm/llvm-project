; RUN: llc < %s -mtriple=arm64-eabi -aarch64-redzone | FileCheck %s

define i32 @foo(i32 %a, i32 %b) nounwind ssp {
; CHECK-LABEL: foo:
; CHECK-NOT: sub sp, sp
; CHECK: ret
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  %tmp = load i32, ptr %a.addr, align 4
  %tmp1 = load i32, ptr %b.addr, align 4
  %add = add nsw i32 %tmp, %tmp1
  store i32 %add, ptr %x, align 4
  %tmp2 = load i32, ptr %x, align 4
  ret i32 %tmp2
}

; We disable red-zone if NEON is available because copies of Q-regs
; require a spill/fill and dynamic allocation. But we only need to do
; this when FP registers are enabled.
define void @bar(fp128 %f) "target-features"="-fp-armv8" {
; CHECK-LABEL: bar:
; CHECK: // %bb.0:
; CHECK-NEXT: stp x0, x1, [sp, #-16]
; CHECK-NEXT: ret
  %ptr = alloca fp128
  store fp128 %f, ptr %ptr
  ret void
}
