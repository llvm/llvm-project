; Test the Test Data Class instruction logic operation folding.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @llvm.s390.tdc.f32(float, i64)
declare i32 @llvm.s390.tdc.f64(double, i64)
declare i32 @llvm.s390.tdc.f128(fp128, i64)

; Check using or i1
define i32 @f1(float %x) {
; CHECK-LABEL: f1
; CHECK: tceb %f0, 7
; CHECK-NEXT: ipm [[REG1:%r[0-9]+]]
; CHECK-NEXT: risbg %r2, [[REG1]], 63, 191, 36
  %a = call i32 @llvm.s390.tdc.f32(float %x, i64 3)
  %b = call i32 @llvm.s390.tdc.f32(float %x, i64 6)
  %a1 = icmp ne i32 %a, 0
  %b1 = icmp ne i32 %b, 0
  %res = or i1 %a1, %b1
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Check using and i1
define i32 @f2(double %x) {
; CHECK-LABEL: f2
; CHECK: tcdb %f0, 2
; CHECK-NEXT: ipm [[REG1:%r[0-9]+]]
; CHECK-NEXT: risbg %r2, [[REG1]], 63, 191, 36
  %a = call i32 @llvm.s390.tdc.f64(double %x, i64 3)
  %b = call i32 @llvm.s390.tdc.f64(double %x, i64 6)
  %a1 = icmp ne i32 %a, 0
  %b1 = icmp ne i32 %b, 0
  %res = and i1 %a1, %b1
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Check using xor i1
define i32 @f3(fp128 %x) {
; CHECK-LABEL: f3
; CHECK: tcxb %f0, 5
; CHECK-NEXT: ipm [[REG1:%r[0-9]+]]
; CHECK-NEXT: risbg %r2, [[REG1]], 63, 191, 36
  %a = call i32 @llvm.s390.tdc.f128(fp128 %x, i64 3)
  %b = call i32 @llvm.s390.tdc.f128(fp128 %x, i64 6)
  %a1 = icmp ne i32 %a, 0
  %b1 = icmp ne i32 %b, 0
  %res = xor i1 %a1, %b1
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Check using xor i1 - negated test
define i32 @f4(fp128 %x) {
; CHECK-LABEL: f4
; CHECK: tcxb %f0, 4090
; CHECK-NEXT: ipm [[REG1:%r[0-9]+]]
; CHECK-NEXT: risbg %r2, [[REG1]], 63, 191, 36
  %a = call i32 @llvm.s390.tdc.f128(fp128 %x, i64 3)
  %b = call i32 @llvm.s390.tdc.f128(fp128 %x, i64 6)
  %a1 = icmp ne i32 %a, 0
  %b1 = icmp eq i32 %b, 0
  %res = xor i1 %a1, %b1
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Check different first args
define i32 @f5(float %x, float %y) {
; CHECK-LABEL: f5
; CHECK-NOT: tceb {{%f[0-9]+}}, 5
; CHECK-DAG: tceb %f0, 3
; CHECK-DAG: tceb %f2, 6
  %a = call i32 @llvm.s390.tdc.f32(float %x, i64 3)
  %b = call i32 @llvm.s390.tdc.f32(float %y, i64 6)
  %a1 = icmp ne i32 %a, 0
  %b1 = icmp ne i32 %b, 0
  %res = xor i1 %a1, %b1
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Non-const mask (not supported)
define i32 @f6(float %x, i64 %y) {
; CHECK-LABEL: f6
; CHECK-DAG: tceb %f0, 0(%r2)
; CHECK-DAG: tceb %f0, 6
  %a = call i32 @llvm.s390.tdc.f32(float %x, i64 %y)
  %b = call i32 @llvm.s390.tdc.f32(float %x, i64 6)
  %a1 = icmp ne i32 %a, 0
  %b1 = icmp ne i32 %b, 0
  %res = xor i1 %a1, %b1
  %xres = zext i1 %res to i32
  ret i32 %xres
}
