; Verify that the PISA target intrinsics are registered in the IR layer and
; round-trip through the verifier/assembler.

; RUN: opt -S < %s | FileCheck %s

define void @test() {
; CHECK-LABEL: define void @test()
  %lane = call i32 @llvm.pisa.lane.id()
; CHECK: call i32 @llvm.pisa.lane.id()
  %sgsize = call i32 @llvm.pisa.subgroup.size()
; CHECK: call i32 @llvm.pisa.subgroup.size()
  %wdim = call i32 @llvm.pisa.work.dim()
; CHECK: call i32 @llvm.pisa.work.dim()
  ret void
}

declare i32 @llvm.pisa.lane.id()
declare i32 @llvm.pisa.subgroup.size()
declare i32 @llvm.pisa.work.dim()
