; Test the Test Data Class instruction, selected manually via the intrinsic.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @llvm.s390.tdc.f32(float, i64)
declare i32 @llvm.s390.tdc.f64(double, i64)
declare i32 @llvm.s390.tdc.f128(fp128, i64)

; Check using as i32 - f32
define i32 @f1(float %x) {
; CHECK-LABEL: f1
; CHECK: tceb %f0, 123
; CHECK: ipm %r2
; CHECK: srl %r2, 28
  %res = call i32 @llvm.s390.tdc.f32(float %x, i64 123)
  ret i32 %res
}

; Check using as i32 - f64
define i32 @f2(double %x) {
; CHECK-LABEL: f2
; CHECK: tcdb %f0, 123
; CHECK: ipm %r2
; CHECK: srl %r2, 28
  %res = call i32 @llvm.s390.tdc.f64(double %x, i64 123)
  ret i32 %res
}

; Check using as i32 - f128
define i32 @f3(fp128 %x) {
; CHECK-LABEL: f3
; CHECK: ld %f0, 0(%r2)
; CHECK: ld %f2, 8(%r2)
; CHECK: tcxb %f0, 123
; CHECK: ipm %r2
; CHECK: srl %r2, 28
  %res = call i32 @llvm.s390.tdc.f128(fp128 %x, i64 123)
  ret i32 %res
}

declare void @g()

; Check branch
define void @f4(float %x) {
; CHECK-LABEL: f4
; CHECK: tceb %f0, 123
; CHECK: jgl g
; CHECK: br %r14
  %res = call i32 @llvm.s390.tdc.f32(float %x, i64 123)
  %cond = icmp ne i32 %res, 0
  br i1 %cond, label %call, label %exit

call:
  tail call void @g()
  br label %exit

exit:
  ret void
}

; Check branch negated
define void @f5(float %x) {
; CHECK-LABEL: f5
; CHECK: tceb %f0, 123
; CHECK: jge g
; CHECK: br %r14
  %res = call i32 @llvm.s390.tdc.f32(float %x, i64 123)
  %cond = icmp eq i32 %res, 0
  br i1 %cond, label %call, label %exit

call:
  tail call void @g()
  br label %exit

exit:
  ret void
}

; Check non-const mask
define void @f6(float %x, i64 %y) {
; CHECK-LABEL: f6
; CHECK: tceb %f0, 0(%r2)
; CHECK: jge g
; CHECK: br %r14
  %res = call i32 @llvm.s390.tdc.f32(float %x, i64 %y)
  %cond = icmp eq i32 %res, 0
  br i1 %cond, label %call, label %exit

call:
  tail call void @g()
  br label %exit

exit:
  ret void
}
