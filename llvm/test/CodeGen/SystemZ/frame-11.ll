; Test the stackrestore builtin.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare ptr@llvm.stacksave()
declare void @llvm.stackrestore(ptr)

; we should use a frame pointer and tear down the frame based on %r11
; rather than %r15.
define void @f1(i32 %count1, i32 %count2) {
; CHECK-LABEL: f1:
; CHECK: stmg %r11, %r15, 88(%r15)
; CHECK: aghi %r15, -160
; CHECK: lgr %r11, %r15
; CHECK: lgr %r15, %r{{[0-5]}}
; CHECK: lmg %r11, %r15, 248(%r11)
; CHECK: br %r14
  %src = call ptr@llvm.stacksave()
  %array1 = alloca i8, i32 %count1
  store volatile i8 0, ptr %array1
  call void @llvm.stackrestore(ptr %src)
  %array2 = alloca i8, i32 %count2
  store volatile i8 0, ptr %array2
  ret void
}
