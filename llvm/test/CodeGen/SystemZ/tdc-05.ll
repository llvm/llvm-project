; Test the Test Data Class instruction logic operation conversion from
; compares, combined with signbit or other compares to ensure worthiness.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
;

declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare fp128 @llvm.fabs.f128(fp128)

; Compare with 0, extract sign bit
define i32 @f0(half %x) {
; CHECK-LABEL: f0
; CHECK:       lgdr %r0, %f0
; CHECK-NEXT:  srlg %r0, %r0, 48
; CHECK-NEXT:  lhr %r0, %r0
; CHECK-NEXT:  chi %r0, 0
; CHECK-NEXT:  ipm %r0
; CHECK-NEXT:  risbg %r13, %r0, 63, 191, 36
; CHECK-NEXT:     # kill: def $f0h killed $f0h killed $f0d
; CHECK-NEXT:  brasl %r14, __extendhfsf2@PLT
; CHECK-NEXT:  ltebr %f0, %f0
; CHECK-NEXT:  ipm %r0
; CHECK-NEXT:  rosbg %r13, %r0, 63, 63, 35
; CHECK-NEXT:  lr %r2, %r13
; CHECK-NEXT:  lmg %r13, %r15, 264(%r15)
; CHECK-NEXT:  br %r14
  %cast = bitcast half %x to i16
  %sign = icmp slt i16 %cast, 0
  %fcmp = fcmp ugt half %x, 0.0
  %res = or i1 %sign, %fcmp
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Compare with 0, extract sign bit
define i32 @f1(float %x) {
; CHECK-LABEL: f1
; CHECK: tceb %f0, 2047
  %cast = bitcast float %x to i32
  %sign = icmp slt i32 %cast, 0
  %fcmp = fcmp ugt float %x, 0.0
  %res = or i1 %sign, %fcmp
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Compare with inf, extract negated sign bit
define i32 @f2(float %x) {
; CHECK-LABEL: f2
; CHECK: tceb %f0, 2698
  %cast = bitcast float %x to i32
  %sign = icmp sgt i32 %cast, -1
  %fcmp = fcmp ult float %x, 0x7ff0000000000000
  %res = and i1 %sign, %fcmp
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Compare with minnorm, extract negated sign bit
define i32 @f3(float %x) {
; CHECK-LABEL: f3
; CHECK: tceb %f0, 2176
  %cast = bitcast float %x to i32
  %sign = icmp sgt i32 %cast, -1
  %fcmp = fcmp olt float %x, 0x3810000000000000
  %res = and i1 %sign, %fcmp
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Test float isnormal, from clang.
define i32 @f4(float %x) {
; CHECK-LABEL: f4
; CHECK: tceb %f0, 768
  %y = call float @llvm.fabs.f32(float %x)
  %ord = fcmp ord float %x, 0.0
  %a = fcmp ult float %y, 0x7ff0000000000000
  %b = fcmp uge float %y, 0x3810000000000000
  %c = and i1 %a, %b
  %res = and i1 %ord, %c
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Check for negative 0.
define i32 @f5(float %x) {
; CHECK-LABEL: f5
; CHECK: tceb %f0, 1024
  %cast = bitcast float %x to i32
  %sign = icmp slt i32 %cast, 0
  %fcmp = fcmp oeq float %x, 0.0
  %res = and i1 %sign, %fcmp
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Test isnormal, from clang.
define i32 @f6(double %x) {
; CHECK-LABEL: f6
; CHECK: tcdb %f0, 768
  %y = call double @llvm.fabs.f64(double %x)
  %ord = fcmp ord double %x, 0.0
  %a = fcmp ult double %y, 0x7ff0000000000000
  %b = fcmp uge double %y, 0x0010000000000000
  %c = and i1 %ord, %a
  %res = and i1 %b, %c
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Test isinf || isnan, from clang.
define i32 @f7(double %x) {
; CHECK-LABEL: f7
; CHECK: tcdb %f0, 63
  %y = call double @llvm.fabs.f64(double %x)
  %a = fcmp oeq double %y, 0x7ff0000000000000
  %b = fcmp uno double %x, 0.0
  %res = or i1 %a, %b
  %xres = zext i1 %res to i32
  ret i32 %xres
}
