; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

declare void @bar(ptr, i64)

; Function Attrs: nounwind
define void @test(i64 %n) {
; CHECK-LABEL: test:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, 15(, %s0)
; CHECK-NEXT:    and %s0, -16, %s0
; CHECK-NEXT:    lea %s2, __ve_grow_stack@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, __ve_grow_stack@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    lea %s0, 240(, %s11)
; CHECK-NEXT:    lea %s2, bar@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, bar@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %dyna = alloca i8, i64 %n, align 8
  call void @bar(ptr %dyna, i64 %n)
  ret void
}
