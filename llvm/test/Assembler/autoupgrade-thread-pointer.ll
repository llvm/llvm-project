; Test autoupgrade of arch-specific thread pointer intrinsics
; RUN: llvm-as < %s | llvm-dis | FileCheck %s

declare i8* @llvm.aarch64.thread.pointer()
declare i8* @llvm.arm.thread.pointer()

define i8* @test1() {
; CHECK: test1()
; CHECK: call i8* @llvm.thread.pointer()
  %1 = call i8* @llvm.aarch64.thread.pointer()
  ret i8 *%1
}

define i8* @test2() {
; CHECK: test2()
; CHECK: call i8* @llvm.thread.pointer()
  %1 = call i8* @llvm.arm.thread.pointer()
  ret i8 *%1
}
