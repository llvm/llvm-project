; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define i32 @test_smin(i32 %arg0, i32 %arg1) {
; CHECK-LABEL: define i32 @test_smin(
; CHECK-NEXT: %v0 = tail call i32 @llvm.smin.i32(i32 %arg0, i32 %arg1)
; CHECK-NEXT: %v1 = add nsw i32 %arg0, 1
; CHECK-NEXT: %v2 = tail call i32 @llvm.smin.i32(i32 %v1, i32 %arg1)
; CHECK-NEXT: %v3 = sub i32 %v2, %v0
; CHECK-NEXT: ret i32 %v3
;
  %v0 = tail call i32 @llvm.smin.i32(i32 %arg0, i32 %arg1)
  %v1 = add nsw i32 %arg0, 1
  %v2 = tail call i32 @llvm.smin.i32(i32 %v1, i32 %arg1)
  %v3 = sub i32 %v2, %v0
  ret i32 %v3
}

define i32 @test_umin(i32 %arg0, i32 %arg1) {
; CHECK-LABEL: define i32 @test_umin(
; CHECK-NEXT: %v0 = tail call i32 @llvm.umin.i32(i32 %arg0, i32 %arg1)
; CHECK-NEXT: %v1 = add nuw i32 %arg0, 1
; CHECK-NEXT: %v2 = tail call i32 @llvm.umin.i32(i32 %v1, i32 %arg1)
; CHECK-NEXT: %v3 = sub i32 %v2, %v0
; CHECK-NEXT: ret i32 %v3
;
  %v0 = tail call i32 @llvm.umin.i32(i32 %arg0, i32 %arg1)
  %v1 = add nuw i32 %arg0, 1
  %v2 = tail call i32 @llvm.umin.i32(i32 %v1, i32 %arg1)
  %v3 = sub i32 %v2, %v0
  ret i32 %v3
}

define i1 @test_smin_i1(i1 %arg0, i1 %arg1) {
; CHECK-LABEL: define i1 @test_smin_i1(
; CHECK-NEXT: %v0 = or i1 %arg0, %arg1
; CHECK-NEXT: %v3 = xor i1 %v0, true
; CHECK-NEXT: ret i1 %v3
;
  %v0 = tail call i1 @llvm.smin.i1(i1 %arg0, i1 %arg1)
  %v1 = add nsw i1 %arg0, 1
  %v2 = tail call i1 @llvm.smin.i1(i1 %v1, i1 %arg1)
  %v3 = sub i1 %v2, %v0
  ret i1 %v3
}

declare void @use(i2)

define i2 @test_smin_use_operands(i2 %arg0, i2 %arg1) {
; CHECK-LABEL: define i2 @test_smin_use_operands(
; CHECK-NEXT: %v0 = tail call i2 @llvm.smin.i2(i2 %arg0, i2 %arg1)
; CHECK-NEXT: %v1 = add nsw i2 %arg0, 1
; CHECK-NEXT: %v2 = tail call i2 @llvm.smin.i2(i2 %v1, i2 %arg1)
; CHECK-NEXT: %v3 = sub i2 %v2, %v0
; CHECK-NEXT: call void @use(i2 %v2)
; CHECK-NEXT: call void @use(i2 %v0)
; CHECK-NEXT: ret i2 %v3
;
  %v0 = tail call i2 @llvm.smin.i2(i2 %arg0, i2 %arg1)
  %v1 = add nsw i2 %arg0, 1
  %v2 = tail call i2 @llvm.smin.i2(i2 %v1, i2 %arg1)
  %v3 = sub i2 %v2, %v0 
  call void @use(i2 %v2)
  call void @use(i2 %v0)
  ret i2 %v3 
}

define i2 @test_smin_use_operand(i2 %arg0, i2 %arg1) {
; CHECK-LABEL: define i2 @test_smin_use_operand(
; CHECK-NEXT: %v0 = tail call i2 @llvm.smin.i2(i2 %arg0, i2 %arg1)
; CHECK-NEXT: %v1 = add nsw i2 %arg0, 1
; CHECK-NEXT: %v2 = tail call i2 @llvm.smin.i2(i2 %v1, i2 %arg1)
; CHECK-NEXT: %v3 = sub i2 %v2, %v0
; CHECK-NEXT: call void @use(i2 %v2)
; CHECK-NEXT: ret i2 %v3
;
  %v0 = tail call i2 @llvm.smin.i2(i2 %arg0, i2 %arg1)
  %v1 = add nsw i2 %arg0, 1
  %v2 = tail call i2 @llvm.smin.i2(i2 %v1, i2 %arg1)
  %v3 = sub i2 %v2, %v0 
  call void @use(i2 %v2)
  ret i2 %v3 
}

define i32 @test_smin_missing_nsw(i32 %arg0, i32 %arg1) {
; CHECK-LABEL: define i32 @test_smin_missing_nsw(
; CHECK-NEXT: %v0 = tail call i32 @llvm.smin.i32(i32 %arg0, i32 %arg1)
; CHECK-NEXT: %v1 = add i32 %arg0, 1
; CHECK-NEXT: %v2 = tail call i32 @llvm.smin.i32(i32 %v1, i32 %arg1)
; CHECK-NEXT: %v3 = sub i32 %v2, %v0
; CHECK-NEXT: ret i32 %v3
;
  %v0 = tail call i32 @llvm.smin.i32(i32 %arg0, i32 %arg1)
  %v1 = add i32 %arg0, 1
  %v2 = tail call i32 @llvm.smin.i32(i32 %v1, i32 %arg1)
  %v3 = sub i32 %v2, %v0
  ret i32 %v3
}

define i32 @test_umin_missing_nuw(i32 %arg0, i32 %arg1) {
; CHECK-LABEL: define i32 @test_umin_missing_nuw(
; CHECK-NEXT: %v0 = tail call i32 @llvm.umin.i32(i32 %arg0, i32 %arg1)
; CHECK-NEXT: %v1 = add i32 %arg0, 1
; CHECK-NEXT: %v2 = tail call i32 @llvm.umin.i32(i32 %v1, i32 %arg1)
; CHECK-NEXT: %v3 = sub i32 %v2, %v0
; CHECK-NEXT: ret i32 %v3
;
  %v0 = tail call i32 @llvm.umin.i32(i32 %arg0, i32 %arg1)
  %v1 = add i32 %arg0, 1
  %v2 = tail call i32 @llvm.umin.i32(i32 %v1, i32 %arg1)
  %v3 = sub i32 %v2, %v0
  ret i32 %v3
}
