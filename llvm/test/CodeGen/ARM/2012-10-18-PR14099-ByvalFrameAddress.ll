; RUN: llc < %s -mtriple=armv7-linux-gnueabi | FileCheck %s

%struct.s = type { [4 x i32] }
@v = constant %struct.s zeroinitializer; 

declare void @f(%struct.s* %p);

; CHECK: t:
define void @t(i32 %a, %struct.s* byval %s) nounwind {
entry:

; Here we need to only check proper start address of restored %s argument.
; CHECK:      sub     sp, sp, #16
; CHECK:      push    {r11, lr}
; CHECK:      add     r0, sp, #12
; CHECK:      stm     r0, {r1, r2, r3}
; CHECK:      add     r0, sp, #12
; CHECK-NEXT: bl f
  call void @f(%struct.s* %s)
  ret void
}

; CHECK: caller:
define void @caller() {

; CHECK:      ldm     r0, {r1, r2, r3}
  call void @t(i32 0, %struct.s* @v);
  ret void
}
