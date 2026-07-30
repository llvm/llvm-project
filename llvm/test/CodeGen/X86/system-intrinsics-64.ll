; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+fxsr | FileCheck %s

define void @test_fxsave(ptr %ptr) {
; CHECK-LABEL: test_fxsave
; CHECK: fxsave
  call void @llvm.x86.fxsave(ptr %ptr)
  ret void;
}
declare void @llvm.x86.fxsave(ptr)

define void @test_fxsave64(ptr %ptr) {
; CHECK-LABEL: test_fxsave64
; CHECK: fxsave64
  call void @llvm.x86.fxsave64(ptr %ptr)
  ret void;
}
declare void @llvm.x86.fxsave64(ptr)

define void @test_fxrstor(ptr %ptr) {
; CHECK-LABEL: test_fxrstor
; CHECK: fxrstor
  call void @llvm.x86.fxrstor(ptr %ptr)
  ret void;
}
declare void @llvm.x86.fxrstor(ptr)

define void @test_fxrstor64(ptr %ptr) {
; CHECK-LABEL: test_fxrstor64
; CHECK: fxrstor64
  call void @llvm.x86.fxrstor64(ptr %ptr)
  ret void;
}
declare void @llvm.x86.fxrstor64(ptr)
