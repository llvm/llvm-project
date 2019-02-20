; RUN: opt -codegenprepare -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; CHECK-LABEL: @test1(
; objectsize should fold to a constant, which causes the branch to fold to an
; uncond branch. Next, we fold the control flow alltogether.
; rdar://8785296
define i32 @test1(i8* %ptr) nounwind ssp noredzone align 2 {
entry:
  %0 = tail call i64 @llvm.objectsize.i64(i8* %ptr, i1 false, i1 false, i1 false)
  %1 = icmp ugt i64 %0, 3
  br i1 %1, label %T, label %trap

; CHECK: entry:
; CHECK-NOT: br label %

trap:                                             ; preds = %0, %entry
  tail call void @llvm.trap() noreturn nounwind
  unreachable

T:
; CHECK: ret i32 4
  ret i32 4
}

; CHECK-LABEL: @test_objectsize_null_flag(
define i64 @test_objectsize_null_flag(i8* %ptr) {
entry:
  ; CHECK: ret i64 -1
  %0 = tail call i64 @llvm.objectsize.i64(i8* null, i1 false, i1 true, i1 false)
  ret i64 %0
}

; CHECK-LABEL: @test_objectsize_null_flag_min(
define i64 @test_objectsize_null_flag_min(i8* %ptr) {
entry:
  ; CHECK: ret i64 0
  %0 = tail call i64 @llvm.objectsize.i64(i8* null, i1 true, i1 true, i1 false)
  ret i64 %0
}

; Test foldable null pointers because we evaluate them with non-exact modes in
; CodeGenPrepare.
; CHECK-LABEL: @test_objectsize_null_flag_noas0(
define i64 @test_objectsize_null_flag_noas0() {
entry:
  ; CHECK: ret i64 -1
  %0 = tail call i64 @llvm.objectsize.i64.p1i8(i8 addrspace(1)* null, i1 false,
                                               i1 true, i1 false)
  ret i64 %0
}

; CHECK-LABEL: @test_objectsize_null_flag_min_noas0(
define i64 @test_objectsize_null_flag_min_noas0() {
entry:
  ; CHECK: ret i64 0
  %0 = tail call i64 @llvm.objectsize.i64.p1i8(i8 addrspace(1)* null, i1 true,
                                               i1 true, i1 false)
  ret i64 %0
}

; CHECK-LABEL: @test_objectsize_null_known_flag_noas0
define i64 @test_objectsize_null_known_flag_noas0() {
entry:
  ; CHECK: ret i64 -1
  %0 = tail call i64 @llvm.objectsize.i64.p1i8(i8 addrspace(1)* null, i1 false,
                                               i1 false, i1 false)
  ret i64 %0
}

; CHECK-LABEL: @test_objectsize_null_known_flag_min_noas0
define i64 @test_objectsize_null_known_flag_min_noas0() {
entry:
  ; CHECK: ret i64 0
  %0 = tail call i64 @llvm.objectsize.i64.p1i8(i8 addrspace(1)* null, i1 true,
                                               i1 false, i1 false)
  ret i64 %0
}


declare i64 @llvm.objectsize.i64(i8*, i1, i1, i1) nounwind readonly
declare i64 @llvm.objectsize.i64.p1i8(i8 addrspace(1)*, i1, i1, i1) nounwind readonly

declare void @llvm.trap() nounwind
