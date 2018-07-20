; Test moves between FPRs and GPRs.  The 32-bit cases test the z10
; implementation, which has no high-word support.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

declare i64 @foo()
declare double @bar()
@dptr = external global double
@iptr = external global i64

; Test 32-bit moves from GPRs to FPRs.  The GPR must be moved into the high
; 32 bits of the FPR.
define float @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: sllg [[REGISTER:%r[0-5]]], %r2, 32
; CHECK: ldgr %f0, [[REGISTER]]
  %res = bitcast i32 %a to float
  ret float %res
}

; Like f1, but create a situation where the shift can be folded with
; surrounding code.
define float @f2(i64 %big) {
; CHECK-LABEL: f2:
; CHECK: risbg [[REGISTER:%r[0-5]]], %r2, 0, 159, 31
; CHECK: ldgr %f0, [[REGISTER]]
  %shift = lshr i64 %big, 1
  %a = trunc i64 %shift to i32
  %res = bitcast i32 %a to float
  ret float %res
}

; Another example of the same thing.
define float @f3(i64 %big) {
; CHECK-LABEL: f3:
; CHECK: risbg [[REGISTER:%r[0-5]]], %r2, 0, 159, 2
; CHECK: ldgr %f0, [[REGISTER]]
  %shift = ashr i64 %big, 30
  %a = trunc i64 %shift to i32
  %res = bitcast i32 %a to float
  ret float %res
}

; Like f1, but the value to transfer is already in the high 32 bits.
define float @f4(i64 %big) {
; CHECK-LABEL: f4:
; CHECK-NOT: %r2
; CHECK: nilf %r2, 0
; CHECK-NOT: %r2
; CHECK: ldgr %f0, %r2
  %shift = ashr i64 %big, 32
  %a = trunc i64 %shift to i32
  %res = bitcast i32 %a to float
  ret float %res
}

; Test 64-bit moves from GPRs to FPRs.
define double @f5(i64 %a) {
; CHECK-LABEL: f5:
; CHECK: ldgr %f0, %r2
  %res = bitcast i64 %a to double
  ret double %res
}

; Test 128-bit moves from GPRs to FPRs.  i128 isn't a legitimate type,
; so this goes through memory.
define void @f6(fp128 *%a, i128 *%b) {
; CHECK-LABEL: f6:
; CHECK: lg
; CHECK: lg
; CHECK: stg
; CHECK: stg
; CHECK: br %r14
  %val = load i128, i128 *%b
  %res = bitcast i128 %val to fp128
  store fp128 %res, fp128 *%a
  ret void
}

; Test 32-bit moves from FPRs to GPRs.  The high 32 bits of the FPR should
; be moved into the low 32 bits of the GPR.
define i32 @f7(float %a) {
; CHECK-LABEL: f7:
; CHECK: lgdr [[REGISTER:%r[0-5]]], %f0
; CHECK: srlg %r2, [[REGISTER]], 32
  %res = bitcast float %a to i32
  ret i32 %res
}

; Test 64-bit moves from FPRs to GPRs.
define i64 @f8(double %a) {
; CHECK-LABEL: f8:
; CHECK: lgdr %r2, %f0
  %res = bitcast double %a to i64
  ret i64 %res
}

; Test 128-bit moves from FPRs to GPRs, with the same restriction as f6.
define void @f9(fp128 *%a, i128 *%b) {
; CHECK-LABEL: f9:
; CHECK: ld
; CHECK: ld
; CHECK: std
; CHECK: std
  %val = load fp128, fp128 *%a
  %res = bitcast fp128 %val to i128
  store i128 %res, i128 *%b
  ret void
}

; Test cases where the destination of an LGDR needs to be spilled.
; We shouldn't have any integer stack stores or floating-point loads.
define void @f10(double %extra) {
; CHECK-LABEL: f10:
; CHECK: dptr
; CHECK-NOT: stg {{.*}}(%r15)
; CHECK: %loop
; CHECK-NOT: ld {{.*}}(%r15)
; CHECK: %exit
; CHECK: br %r14
entry:
  %double0 = load volatile double, double *@dptr
  %biased0 = fadd double %double0, %extra
  %int0 = bitcast double %biased0 to i64
  %double1 = load volatile double, double *@dptr
  %biased1 = fadd double %double1, %extra
  %int1 = bitcast double %biased1 to i64
  %double2 = load volatile double, double *@dptr
  %biased2 = fadd double %double2, %extra
  %int2 = bitcast double %biased2 to i64
  %double3 = load volatile double, double *@dptr
  %biased3 = fadd double %double3, %extra
  %int3 = bitcast double %biased3 to i64
  %double4 = load volatile double, double *@dptr
  %biased4 = fadd double %double4, %extra
  %int4 = bitcast double %biased4 to i64
  %double5 = load volatile double, double *@dptr
  %biased5 = fadd double %double5, %extra
  %int5 = bitcast double %biased5 to i64
  %double6 = load volatile double, double *@dptr
  %biased6 = fadd double %double6, %extra
  %int6 = bitcast double %biased6 to i64
  %double7 = load volatile double, double *@dptr
  %biased7 = fadd double %double7, %extra
  %int7 = bitcast double %biased7 to i64
  %double8 = load volatile double, double *@dptr
  %biased8 = fadd double %double8, %extra
  %int8 = bitcast double %biased8 to i64
  %double9 = load volatile double, double *@dptr
  %biased9 = fadd double %double9, %extra
  %int9 = bitcast double %biased9 to i64
  br label %loop

loop:
  %start = call i64 @foo()
  %or0 = or i64 %start, %int0
  %or1 = or i64 %or0, %int1
  %or2 = or i64 %or1, %int2
  %or3 = or i64 %or2, %int3
  %or4 = or i64 %or3, %int4
  %or5 = or i64 %or4, %int5
  %or6 = or i64 %or5, %int6
  %or7 = or i64 %or6, %int7
  %or8 = or i64 %or7, %int8
  %or9 = or i64 %or8, %int9
  store i64 %or9, i64 *@iptr
  %cont = icmp ne i64 %start, 1
  br i1 %cont, label %loop, label %exit

exit:
  ret void
}

; ...likewise LDGR, with the requirements the other way around.
define void @f11(i64 %mask) {
; CHECK-LABEL: f11:
; CHECK: iptr
; CHECK-NOT: std {{.*}}(%r15)
; CHECK: %loop
; CHECK-NOT: lg {{.*}}(%r15)
; CHECK: %exit
; CHECK: br %r14
entry:
  %int0 = load volatile i64, i64 *@iptr
  %masked0 = and i64 %int0, %mask
  %double0 = bitcast i64 %masked0 to double
  %int1 = load volatile i64, i64 *@iptr
  %masked1 = and i64 %int1, %mask
  %double1 = bitcast i64 %masked1 to double
  %int2 = load volatile i64, i64 *@iptr
  %masked2 = and i64 %int2, %mask
  %double2 = bitcast i64 %masked2 to double
  %int3 = load volatile i64, i64 *@iptr
  %masked3 = and i64 %int3, %mask
  %double3 = bitcast i64 %masked3 to double
  %int4 = load volatile i64, i64 *@iptr
  %masked4 = and i64 %int4, %mask
  %double4 = bitcast i64 %masked4 to double
  %int5 = load volatile i64, i64 *@iptr
  %masked5 = and i64 %int5, %mask
  %double5 = bitcast i64 %masked5 to double
  %int6 = load volatile i64, i64 *@iptr
  %masked6 = and i64 %int6, %mask
  %double6 = bitcast i64 %masked6 to double
  %int7 = load volatile i64, i64 *@iptr
  %masked7 = and i64 %int7, %mask
  %double7 = bitcast i64 %masked7 to double
  %int8 = load volatile i64, i64 *@iptr
  %masked8 = and i64 %int8, %mask
  %double8 = bitcast i64 %masked8 to double
  %int9 = load volatile i64, i64 *@iptr
  %masked9 = and i64 %int9, %mask
  %double9 = bitcast i64 %masked9 to double
  br label %loop

loop:
  %start = call double @bar()
  %add0 = fadd double %start, %double0
  %add1 = fadd double %add0, %double1
  %add2 = fadd double %add1, %double2
  %add3 = fadd double %add2, %double3
  %add4 = fadd double %add3, %double4
  %add5 = fadd double %add4, %double5
  %add6 = fadd double %add5, %double6
  %add7 = fadd double %add6, %double7
  %add8 = fadd double %add7, %double8
  %add9 = fadd double %add8, %double9
  store double %add9, double *@dptr
  %cont = fcmp one double %start, 1.0
  br i1 %cont, label %loop, label %exit

exit:
  ret void
}

; Test cases where the source of an LDGR needs to be spilled.
; We shouldn't have any integer stack stores or floating-point loads.
define void @f12() {
; CHECK-LABEL: f12:
; CHECK: %loop
; CHECK-NOT: std {{.*}}(%r15)
; CHECK: %exit
; CHECK: foo@PLT
; CHECK-NOT: lg {{.*}}(%r15)
; CHECK: foo@PLT
; CHECK: br %r14
entry:
  br label %loop

loop:
  %int0 = phi i64 [ 0, %entry ], [ %add0, %loop ]
  %int1 = phi i64 [ 0, %entry ], [ %add1, %loop ]
  %int2 = phi i64 [ 0, %entry ], [ %add2, %loop ]
  %int3 = phi i64 [ 0, %entry ], [ %add3, %loop ]
  %int4 = phi i64 [ 0, %entry ], [ %add4, %loop ]
  %int5 = phi i64 [ 0, %entry ], [ %add5, %loop ]
  %int6 = phi i64 [ 0, %entry ], [ %add6, %loop ]
  %int7 = phi i64 [ 0, %entry ], [ %add7, %loop ]
  %int8 = phi i64 [ 0, %entry ], [ %add8, %loop ]
  %int9 = phi i64 [ 0, %entry ], [ %add9, %loop ]

  %bias = call i64 @foo()
  %add0 = add i64 %int0, %bias
  %add1 = add i64 %int1, %bias
  %add2 = add i64 %int2, %bias
  %add3 = add i64 %int3, %bias
  %add4 = add i64 %int4, %bias
  %add5 = add i64 %int5, %bias
  %add6 = add i64 %int6, %bias
  %add7 = add i64 %int7, %bias
  %add8 = add i64 %int8, %bias
  %add9 = add i64 %int9, %bias
  %cont = icmp ne i64 %bias, 1
  br i1 %cont, label %loop, label %exit

exit:
  %unused1 = call i64 @foo()
  %factor = load volatile double, double *@dptr

  %conv0 = bitcast i64 %add0 to double
  %mul0 = fmul double %conv0, %factor
  store volatile double %mul0, double *@dptr
  %conv1 = bitcast i64 %add1 to double
  %mul1 = fmul double %conv1, %factor
  store volatile double %mul1, double *@dptr
  %conv2 = bitcast i64 %add2 to double
  %mul2 = fmul double %conv2, %factor
  store volatile double %mul2, double *@dptr
  %conv3 = bitcast i64 %add3 to double
  %mul3 = fmul double %conv3, %factor
  store volatile double %mul3, double *@dptr
  %conv4 = bitcast i64 %add4 to double
  %mul4 = fmul double %conv4, %factor
  store volatile double %mul4, double *@dptr
  %conv5 = bitcast i64 %add5 to double
  %mul5 = fmul double %conv5, %factor
  store volatile double %mul5, double *@dptr
  %conv6 = bitcast i64 %add6 to double
  %mul6 = fmul double %conv6, %factor
  store volatile double %mul6, double *@dptr
  %conv7 = bitcast i64 %add7 to double
  %mul7 = fmul double %conv7, %factor
  store volatile double %mul7, double *@dptr
  %conv8 = bitcast i64 %add8 to double
  %mul8 = fmul double %conv8, %factor
  store volatile double %mul8, double *@dptr
  %conv9 = bitcast i64 %add9 to double
  %mul9 = fmul double %conv9, %factor
  store volatile double %mul9, double *@dptr

  %unused2 = call i64 @foo()

  ret void
}

; ...likewise LGDR, with the requirements the other way around.
define void @f13() {
; CHECK-LABEL: f13:
; CHECK: %loop
; CHECK-NOT: stg {{.*}}(%r15)
; CHECK: %exit
; CHECK: foo@PLT
; CHECK-NOT: ld {{.*}}(%r15)
; CHECK: foo@PLT
; CHECK: br %r14
entry:
  br label %loop

loop:
  %double0 = phi double [ 1.0, %entry ], [ %mul0, %loop ]
  %double1 = phi double [ 1.0, %entry ], [ %mul1, %loop ]
  %double2 = phi double [ 1.0, %entry ], [ %mul2, %loop ]
  %double3 = phi double [ 1.0, %entry ], [ %mul3, %loop ]
  %double4 = phi double [ 1.0, %entry ], [ %mul4, %loop ]
  %double5 = phi double [ 1.0, %entry ], [ %mul5, %loop ]
  %double6 = phi double [ 1.0, %entry ], [ %mul6, %loop ]
  %double7 = phi double [ 1.0, %entry ], [ %mul7, %loop ]
  %double8 = phi double [ 1.0, %entry ], [ %mul8, %loop ]
  %double9 = phi double [ 1.0, %entry ], [ %mul9, %loop ]

  %factor = call double @bar()
  %mul0 = fmul double %double0, %factor
  %mul1 = fmul double %double1, %factor
  %mul2 = fmul double %double2, %factor
  %mul3 = fmul double %double3, %factor
  %mul4 = fmul double %double4, %factor
  %mul5 = fmul double %double5, %factor
  %mul6 = fmul double %double6, %factor
  %mul7 = fmul double %double7, %factor
  %mul8 = fmul double %double8, %factor
  %mul9 = fmul double %double9, %factor
  %cont = fcmp one double %factor, 1.0
  br i1 %cont, label %loop, label %exit

exit:
  %unused1 = call i64 @foo()
  %bias = load volatile i64, i64 *@iptr

  %conv0 = bitcast double %mul0 to i64
  %add0 = add i64 %conv0, %bias
  store volatile i64 %add0, i64 *@iptr
  %conv1 = bitcast double %mul1 to i64
  %add1 = add i64 %conv1, %bias
  store volatile i64 %add1, i64 *@iptr
  %conv2 = bitcast double %mul2 to i64
  %add2 = add i64 %conv2, %bias
  store volatile i64 %add2, i64 *@iptr
  %conv3 = bitcast double %mul3 to i64
  %add3 = add i64 %conv3, %bias
  store volatile i64 %add3, i64 *@iptr
  %conv4 = bitcast double %mul4 to i64
  %add4 = add i64 %conv4, %bias
  store volatile i64 %add4, i64 *@iptr
  %conv5 = bitcast double %mul5 to i64
  %add5 = add i64 %conv5, %bias
  store volatile i64 %add5, i64 *@iptr
  %conv6 = bitcast double %mul6 to i64
  %add6 = add i64 %conv6, %bias
  store volatile i64 %add6, i64 *@iptr
  %conv7 = bitcast double %mul7 to i64
  %add7 = add i64 %conv7, %bias
  store volatile i64 %add7, i64 *@iptr
  %conv8 = bitcast double %mul8 to i64
  %add8 = add i64 %conv8, %bias
  store volatile i64 %add8, i64 *@iptr
  %conv9 = bitcast double %mul9 to i64
  %add9 = add i64 %conv9, %bias
  store volatile i64 %add9, i64 *@iptr

  %unused2 = call i64 @foo()

  ret void
}
