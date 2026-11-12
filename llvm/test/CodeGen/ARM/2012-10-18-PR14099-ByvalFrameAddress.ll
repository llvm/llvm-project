; RUN: llc < %s -mtriple=armv7-linux-gnueabi | FileCheck %s

%struct.s = type { [4 x i32] }
@v = constant %struct.s zeroinitializer;

declare void @f(ptr %p);

; CHECK-LABEL: t:
define void @t(i32 %a, ptr byval(%struct.s) %s) nounwind {
entry:

; Here we need to only check proper start address of restored %s argument.
; CHECK:      sub     sp, sp, #12
; CHECK:      push    {r11, lr}
; CHECK:      sub     sp, sp, #4
; CHECK:      add     r0, sp, #12
; CHECK:      stm     r0, {r1, r2, r3}
; CHECK:      add     r0, sp, #12
; CHECK-NEXT: bl f
  call void @f(ptr %s)
  ret void
}

; CHECK-LABEL: caller:
define void @caller() {

; CHECK:      ldm     r{{[0-9]+}}, {r1, r2, r3}
  call void @t(i32 0, ptr byval(%struct.s) @v);
  ret void
}
