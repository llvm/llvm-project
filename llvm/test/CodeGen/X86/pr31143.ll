; RUN: llc -mtriple=x86_64-pc-linux-gnu -mattr=+sse4.2 < %s | FileCheck %s

; CHECK-LABEL: testss:
; CHECK: movss {{.*}}, %[[XMM0:xmm[0-9]+]]
; CHECK: xorps %[[XMM1:xmm[0-9]+]], %[[XMM1]]
; CHECK: roundss $9, %[[XMM0]], %[[XMM1]]

define void @testss(float* nocapture %a, <4 x float>* nocapture %b, i32 %k) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %indvars.iv
  %v = load float, float* %arrayidx, align 4
  %floor = call float @floorf(float %v)
  %sub = fsub float %floor, %v
  %v1 = insertelement <4 x float> undef, float %sub, i32 0
  %br = shufflevector <4 x float> %v1, <4 x float> undef, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  store volatile <4 x float> %br, <4 x float>* %b, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %k
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; CHECK-LABEL: testsd:
; CHECK: movsd {{.*}}, %[[XMM0:xmm[0-9]+]]
; CHECK: xorps %[[XMM1:xmm[0-9]+]], %[[XMM1]]
; CHECK: roundsd $9, %[[XMM0]], %[[XMM1]]

define void @testsd(double* nocapture %a, <2 x double>* nocapture %b, i32 %k) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %a, i64 %indvars.iv
  %v = load double, double* %arrayidx, align 4
  %floor = call double @floor(double %v)
  %sub = fsub double %floor, %v
  %v1 = insertelement <2 x double> undef, double %sub, i32 0
  %br = shufflevector <2 x double> %v1, <2 x double> undef, <2 x i32> <i32 0, i32 0>
  store volatile <2 x double> %br, <2 x double>* %b, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %k
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare float @floorf(float) nounwind readnone

declare double @floor(double) nounwind readnone

