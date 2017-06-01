; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s

%0 = type { i8, i16, i32, i64, double, i32, [4 x i8] }
%struct.S1 = type { i8, i16, i32, i64, double, i32 }
%struct.S2 = type { [4 x i32] }
%struct.S3 = type { i8 }

@f1.s1 = internal unnamed_addr constant %0 { i8 1, i16 2, i32 3, i64 4, double 5.000000e+00, i32 6, [4 x i8] undef }, align 8
@f1.s2 = internal unnamed_addr constant %struct.S2 { [4 x i32] [i32 7, i32 8, i32 9, i32 10] }, align 4

define void @f1() nounwind {
entry:
; CHECK-LABEL: f1:
; CHECK-DAG: lw  $[[R1:[0-9]+]], %got(f1.s1)
; CHECK-DAG: addiu $[[R0:[0-9]+]], $[[R1]], %lo(f1.s1)
; CHECK-DAG: lw  $[[R7:[0-9]+]], 12($[[R0]])
; CHECK-DAG: lw  $[[R3:[0-9]+]], 16($[[R0]])
; CHECK-DAG: lw  $[[R4:[0-9]+]], 20($[[R0]])
; CHECK-DAG: lw  $[[R5:[0-9]+]], 24($[[R0]])
; CHECK-DAG: lw  $[[R6:[0-9]+]], 28($[[R0]])
; CHECK-DAG: sw  $[[R6]], 36($sp)
; CHECK-DAG: sw  $[[R5]], 32($sp)
; CHECK-DAG: sw  $[[R4]], 28($sp)
; CHECK-DAG: sw  $[[R3]], 24($sp)
; CHECK-DAG: sw  $[[R7]], 20($sp)
; CHECK-DAG: lw  $[[R2:[0-9]+]], 8($[[R0]])
; CHECK-DAG: sw  $[[R2]], 16($sp)
; CHECK-DAG: lw  $6, %lo(f1.s1)($[[R1]])
; CHECK-DAG: lw  $7, 4($[[R0]])
  %agg.tmp10 = alloca %struct.S3, align 4
  call void @callee1(float 2.000000e+01, %struct.S1* byval bitcast (%0* @f1.s1 to %struct.S1*)) nounwind
  call void @callee2(%struct.S2* byval @f1.s2) nounwind
  %tmp11 = getelementptr inbounds %struct.S3, %struct.S3* %agg.tmp10, i32 0, i32 0
  store i8 11, i8* %tmp11, align 4
  call void @callee3(float 2.100000e+01, %struct.S3* byval %agg.tmp10, %struct.S1* byval bitcast (%0* @f1.s1 to %struct.S1*)) nounwind
  ret void
}

declare void @callee1(float, %struct.S1* byval)

declare void @callee2(%struct.S2* byval)

declare void @callee3(float, %struct.S3* byval, %struct.S1* byval)

define void @f2(float %f, %struct.S1* nocapture byval %s1) nounwind {
entry:
; CHECK: addiu $sp, $sp, -48
; CHECK: sw  $7, 60($sp)
; CHECK: sw  $6, 56($sp)
; CHECK: lw  $4, 80($sp)
; CHECK: ldc1 $f[[F0:[0-9]+]], 72($sp)
; CHECK: lw  $[[R3:[0-9]+]], 64($sp)
; CHECK: lw  $[[R4:[0-9]+]], 68($sp)
; CHECK: lw  $[[R2:[0-9]+]], 60($sp)
; CHECK: lh  $[[R1:[0-9]+]], 58($sp)
; CHECK: lb  $[[R0:[0-9]+]], 56($sp)
; CHECK: sw  $[[R0]], 32($sp)
; CHECK: sw  $[[R1]], 28($sp)
; CHECK: sw  $[[R2]], 24($sp)
; CHECK: sw  $[[R4]], 20($sp)
; CHECK: sw  $[[R3]], 16($sp)
; CHECK: mfc1 $6, $f[[F0]]

  %i2 = getelementptr inbounds %struct.S1, %struct.S1* %s1, i32 0, i32 5
  %tmp = load i32, i32* %i2, align 4
  %d = getelementptr inbounds %struct.S1, %struct.S1* %s1, i32 0, i32 4
  %tmp1 = load double, double* %d, align 8
  %ll = getelementptr inbounds %struct.S1, %struct.S1* %s1, i32 0, i32 3
  %tmp2 = load i64, i64* %ll, align 8
  %i = getelementptr inbounds %struct.S1, %struct.S1* %s1, i32 0, i32 2
  %tmp3 = load i32, i32* %i, align 4
  %s = getelementptr inbounds %struct.S1, %struct.S1* %s1, i32 0, i32 1
  %tmp4 = load i16, i16* %s, align 2
  %c = getelementptr inbounds %struct.S1, %struct.S1* %s1, i32 0, i32 0
  %tmp5 = load i8, i8* %c, align 1
  tail call void @callee4(i32 %tmp, double %tmp1, i64 %tmp2, i32 %tmp3, i16 signext %tmp4, i8 signext %tmp5, float %f) nounwind
  ret void
}

declare void @callee4(i32, double, i64, i32, i16 signext, i8 signext, float)

define void @f3(%struct.S2* nocapture byval %s2) nounwind {
entry:
; CHECK: addiu $sp, $sp, -48
; CHECK: sw  $7, 60($sp)
; CHECK: sw  $6, 56($sp)
; CHECK: sw  $5, 52($sp)
; CHECK: sw  $4, 48($sp)
; CHECK: lw  $4, 48($sp)
; CHECK: lw  $[[R0:[0-9]+]], 60($sp)
; CHECK: sw  $[[R0]], 24($sp)

  %arrayidx = getelementptr inbounds %struct.S2, %struct.S2* %s2, i32 0, i32 0, i32 0
  %tmp = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds %struct.S2, %struct.S2* %s2, i32 0, i32 0, i32 3
  %tmp3 = load i32, i32* %arrayidx2, align 4
  tail call void @callee4(i32 %tmp, double 2.000000e+00, i64 3, i32 %tmp3, i16 signext 4, i8 signext 5, float 6.000000e+00) nounwind
  ret void
}

define void @f4(float %f, %struct.S3* nocapture byval %s3, %struct.S1* nocapture byval %s1) nounwind {
entry:
; CHECK: addiu $sp, $sp, -48
; CHECK: sw  $7, 60($sp)
; CHECK: sw  $6, 56($sp)
; CHECK: sw  $5, 52($sp)
; CHECK: lw  $4, 60($sp)
; CHECK: lw  $[[R1:[0-9]+]], 80($sp)
; CHECK: lb  $[[R0:[0-9]+]], 52($sp)
; CHECK: sw  $[[R0]], 32($sp)
; CHECK: sw  $[[R1]], 24($sp)

  %i = getelementptr inbounds %struct.S1, %struct.S1* %s1, i32 0, i32 2
  %tmp = load i32, i32* %i, align 4
  %i2 = getelementptr inbounds %struct.S1, %struct.S1* %s1, i32 0, i32 5
  %tmp1 = load i32, i32* %i2, align 4
  %c = getelementptr inbounds %struct.S3, %struct.S3* %s3, i32 0, i32 0
  %tmp2 = load i8, i8* %c, align 1
  tail call void @callee4(i32 %tmp, double 2.000000e+00, i64 3, i32 %tmp1, i16 signext 4, i8 signext %tmp2, float 6.000000e+00) nounwind
  ret void
}

%struct.S4 = type { [4 x i32] }

define void @f5(i64 %a0, %struct.S4* nocapture byval %a1) nounwind {
entry:
  tail call void @f6(%struct.S4* byval %a1, i64 %a0) nounwind
  ret void
}

declare void @f6(%struct.S4* nocapture byval, i64)
