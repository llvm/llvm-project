; RUN: llc < %s -mtriple=xcore | FileCheck %s

declare void @g()
declare ptr @llvm.stacksave() nounwind
declare void @llvm.stackrestore(ptr) nounwind

define void @f(ptr %p, i32 %size) {
allocas:
  %0 = call ptr @llvm.stacksave()
  %a = alloca i32, i32 %size
  store ptr %a, ptr %p
  call void @g()
  call void @llvm.stackrestore(ptr %0)
  ret void
}
; CHECK-LABEL: f:
; CHECK: ldaw [[REGISTER:r[0-9]+]], {{r[0-9]+}}[-r1]
; CHECK: set sp, [[REGISTER]]
; CHECK: extsp 1
; CHECK: bl g
