; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

declare void @bar(ptr, i64)

; Function Attrs: nounwind
define void @test(i64 %n) {
; CHECK-LABEL: test:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s2, 0, %s0
; CHECK-NEXT:    lea %s0, 15(, %s0)
; CHECK-NEXT:    and %s0, -16, %s0
; CHECK-NEXT:    lea %s1, __ve_grow_stack_align@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, __ve_grow_stack_align@hi(, %s1)
; CHECK-NEXT:    or %s1, -32, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    lea %s0, 240(, %s11)
; CHECK-NEXT:    lea %s0, 31(, %s0)
; CHECK-NEXT:    and %s0, -32, %s0
; CHECK-NEXT:    lea %s1, bar@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, bar@hi(, %s1)
; CHECK-NEXT:    or %s1, 0, %s2
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %dyna = alloca i8, i64 %n, align 32
  call void @bar(ptr %dyna, i64 %n)
  ret void
}
