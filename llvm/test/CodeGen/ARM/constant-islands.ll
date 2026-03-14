; RUN: llc -mtriple=thumbv7-linux-gnueabihf -O0 -fast-isel=0 -o - %s | FileCheck %s

define void @test_no_duplicate_branches(float %in) {
; CHECK-LABEL: test_no_duplicate_branches:
; CHECK: vldr {{s[0-9]+}}, [[CONST:\.LCPI[0-9]+_[0-9]+]]
; CHECK: b .LBB
; CHECK-NOT: b .LBB
; CHECK: [[CONST]]:
; CHECK-NEXT: .long 0x449a5000

  %tst = fcmp oeq float %in, 1234.5

  %chain = zext i1 %tst to i32

  br i1 %tst, label %true, label %false

true:
  call i32 @llvm.arm.space(i32 2000, i32 undef)
  ret void

false:
  ret void
}

declare i32 @llvm.arm.space(i32, i32)
