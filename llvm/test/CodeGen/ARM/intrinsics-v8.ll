; RUN: llc < %s -mtriple=armv8 -mattr=+db | FileCheck %s

define void @test() {
  ; CHECK: dmb sy
  call void @llvm.arm.dmb(i32 15)
  ; CHECK: dmb osh
  call void @llvm.arm.dmb(i32 3)
  ; CHECK: dsb sy
  call void @llvm.arm.dsb(i32 15)
  ; CHECK: dsb ishld
  call void @llvm.arm.dsb(i32 9)
  ; CHECK: sevl
  tail call void @llvm.arm.hint(i32 5) nounwind
  ret void
}

declare void @llvm.arm.dmb(i32)
declare void @llvm.arm.dsb(i32)
declare void @llvm.arm.hint(i32) nounwind
