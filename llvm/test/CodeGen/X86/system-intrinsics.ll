; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+fxsr  | FileCheck %s

define void @test_fxsave(ptr %ptr) {
; CHECK-LABEL: test_fxsave
; CHECK: fxsave
  call void @llvm.x86.fxsave(ptr %ptr)
  ret void;
}
declare void @llvm.x86.fxsave(ptr)

define void @test_fxrstor(ptr %ptr) {
; CHECK-LABEL: test_fxrstor
; CHECK: fxrstor
  call void @llvm.x86.fxrstor(ptr %ptr)
  ret void;
}
declare void @llvm.x86.fxrstor(ptr)
