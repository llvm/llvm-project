; RUN: llc < %s -march=xcore | FileCheck %s

declare void @g()
declare i8* @llvm.stacksave() nounwind
declare void @llvm.stackrestore(i8*) nounwind

define void @f(i32** %p, i32 %size) {
allocas:
  %0 = call i8* @llvm.stacksave()
  %a = alloca i32, i32 %size
  store i32* %a, i32** %p
  call void @g()
  call void @llvm.stackrestore(i8* %0)
  ret void
}
; CHECK: f:
; CHECK: ldaw [[REGISTER:r[0-9]+]], {{r[0-9]+}}[-r1]
; CHECK: set sp, [[REGISTER]]
; CHECK: extsp 1
; CHECK: bl g
