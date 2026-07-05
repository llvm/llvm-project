; RUN: llc < %s -mtriple=thumbv7-linux-gnueabihf %s -o - | FileCheck %s

; Check that new water is created by splitting the basic block after the
; load instruction. Previously, new water was created before the load
; instruction, which caused the pass to fail to converge.

define void @test(i1 %tst) {
; CHECK-LABEL: test:
; CHECK: vldr  {{s[0-9]+}}, [[CONST:\.LCPI[0-9]+_[0-9]+]]

; CHECK: b.w [[CONTINUE:\.LBB[0-9]+_[0-9]+]]
; CHECK: [[CONST]]:
; CHECK-NEXT: .long

; CHECK: [[CONTINUE]]:

entry:
  br i1 %tst, label %true, label %false

true:
  %val = phi float [12345.0, %entry], [undef, %false]
  call i32 @llvm.arm.space(i32 2000, i32 undef)
  call void @bar(float %val)
  ret void

false:
  br label %true
}

declare void @bar(float)
declare i32 @llvm.arm.space(i32, i32)
