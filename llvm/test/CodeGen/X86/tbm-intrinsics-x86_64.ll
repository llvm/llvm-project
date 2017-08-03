; RUN: llc -mtriple=x86_64-unknown-unknown -march=x86-64 -mattr=+tbm < %s | FileCheck %s

define i32 @test_x86_tbm_bextri_u32(i32 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_bextri_u32:
  ; CHECK-NOT: mov
  ; CHECK: bextr $
  %0 = tail call i32 @llvm.x86.tbm.bextri.u32(i32 %a, i32 2814)
  ret i32 %0
}

declare i32 @llvm.x86.tbm.bextri.u32(i32, i32) nounwind readnone

define i32 @test_x86_tbm_bextri_u32_m(i32* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_bextri_u32_m:
  ; CHECK-NOT: mov
  ; CHECK: bextr $
  %tmp1 = load i32, i32* %a, align 4
  %0 = tail call i32 @llvm.x86.tbm.bextri.u32(i32 %tmp1, i32 2814)
  ret i32 %0
}

define i64 @test_x86_tbm_bextri_u64(i64 %a) nounwind readnone {
entry:
  ; CHECK-LABEL: test_x86_tbm_bextri_u64:
  ; CHECK-NOT: mov
  ; CHECK: bextr $
  %0 = tail call i64 @llvm.x86.tbm.bextri.u64(i64 %a, i64 2814)
  ret i64 %0
}

declare i64 @llvm.x86.tbm.bextri.u64(i64, i64) nounwind readnone

define i64 @test_x86_tbm_bextri_u64_m(i64* nocapture %a) nounwind readonly {
entry:
  ; CHECK-LABEL: test_x86_tbm_bextri_u64_m:
  ; CHECK-NOT: mov
  ; CHECK: bextr $
  %tmp1 = load i64, i64* %a, align 8
  %0 = tail call i64 @llvm.x86.tbm.bextri.u64(i64 %tmp1, i64 2814)
  ret i64 %0
}
