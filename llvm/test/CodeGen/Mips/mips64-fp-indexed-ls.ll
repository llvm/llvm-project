; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=n64 < %s | FileCheck %s

%struct.S = type <{ [4 x float] }>
%struct.S2 = type <{ [4 x double] }>
%struct.S3 = type <{ i8, float }>

@s = external global [4 x %struct.S]
@gf = external global float
@gd = external global double
@s2 = external global [4 x %struct.S2]
@s3 = external global %struct.S3

define float @foo0(float* nocapture %b, i32 %o) nounwind readonly {
entry:
; CHECK: lwxc1
  %idxprom = zext i32 %o to i64
  %arrayidx = getelementptr inbounds float* %b, i64 %idxprom
  %0 = load float* %arrayidx, align 4
  ret float %0
}

define double @foo1(double* nocapture %b, i32 %o) nounwind readonly {
entry:
; CHECK: ldxc1
  %idxprom = zext i32 %o to i64
  %arrayidx = getelementptr inbounds double* %b, i64 %idxprom
  %0 = load double* %arrayidx, align 8
  ret double %0
}

define float @foo2(i32 %b, i32 %c) nounwind readonly {
entry:
; CHECK-NOT: luxc1
  %idxprom = zext i32 %c to i64
  %idxprom1 = zext i32 %b to i64
  %arrayidx2 = getelementptr inbounds [4 x %struct.S]* @s, i64 0, i64 %idxprom1, i32 0, i64 %idxprom
  %0 = load float* %arrayidx2, align 1
  ret float %0
}

define void @foo3(float* nocapture %b, i32 %o) nounwind {
entry:
; CHECK: swxc1
  %0 = load float* @gf, align 4
  %idxprom = zext i32 %o to i64
  %arrayidx = getelementptr inbounds float* %b, i64 %idxprom
  store float %0, float* %arrayidx, align 4
  ret void
}

define void @foo4(double* nocapture %b, i32 %o) nounwind {
entry:
; CHECK: sdxc1
  %0 = load double* @gd, align 8
  %idxprom = zext i32 %o to i64
  %arrayidx = getelementptr inbounds double* %b, i64 %idxprom
  store double %0, double* %arrayidx, align 8
  ret void
}

define void @foo5(i32 %b, i32 %c) nounwind {
entry:
; CHECK-NOT: suxc1
  %0 = load float* @gf, align 4
  %idxprom = zext i32 %c to i64
  %idxprom1 = zext i32 %b to i64
  %arrayidx2 = getelementptr inbounds [4 x %struct.S]* @s, i64 0, i64 %idxprom1, i32 0, i64 %idxprom
  store float %0, float* %arrayidx2, align 1
  ret void
}

define double @foo6(i32 %b, i32 %c) nounwind readonly {
entry:
; CHECK: foo6
; CHECK-NOT: luxc1
  %idxprom = zext i32 %c to i64
  %idxprom1 = zext i32 %b to i64
  %arrayidx2 = getelementptr inbounds [4 x %struct.S2]* @s2, i64 0, i64 %idxprom1, i32 0, i64 %idxprom
  %0 = load double* %arrayidx2, align 1
  ret double %0
}

define void @foo7(i32 %b, i32 %c) nounwind {
entry:
; CHECK: foo7
; CHECK-NOT: suxc1
  %0 = load double* @gd, align 8
  %idxprom = zext i32 %c to i64
  %idxprom1 = zext i32 %b to i64
  %arrayidx2 = getelementptr inbounds [4 x %struct.S2]* @s2, i64 0, i64 %idxprom1, i32 0, i64 %idxprom
  store double %0, double* %arrayidx2, align 1
  ret void
}

define float @foo8() nounwind readonly {
entry:
; CHECK: foo8
; CHECK-NOT: luxc1
  %0 = load float* getelementptr inbounds (%struct.S3* @s3, i64 0, i32 1), align 1
  ret float %0
}

define void @foo9(float %f) nounwind {
entry:
; CHECK: foo9
; CHECK-NOT: suxc1
  store float %f, float* getelementptr inbounds (%struct.S3* @s3, i64 0, i32 1), align 1
  ret void
}

