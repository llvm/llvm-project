; RUN: opt < %s -instcombine -S | FileCheck %s

%"struct.std::complex" = type { { float, float } }
@dd = external global %"struct.std::complex", align 4
@dd2 = external global %"struct.std::complex", align 4

define void @_Z3fooi(i32 signext %n) {
entry:
  br label %for.cond

for.cond:
  %ldd.sroa.0.0 = phi i32 [ 0, %entry ], [ %5, %for.body ]
  %ldd.sroa.6.0 = phi i32 [ 0, %entry ], [ %7, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %0 = load float, float* getelementptr inbounds (%"struct.std::complex", %"struct.std::complex"* @dd, i64 0, i32 0, i32 0), align 4
  %1 = load float, float* getelementptr inbounds (%"struct.std::complex", %"struct.std::complex"* @dd, i64 0, i32 0, i32 1), align 4
  %2 = load float, float* getelementptr inbounds (%"struct.std::complex", %"struct.std::complex"* @dd2, i64 0, i32 0, i32 0), align 4
  %3 = load float, float* getelementptr inbounds (%"struct.std::complex", %"struct.std::complex"* @dd2, i64 0, i32 0, i32 1), align 4
  %mul.i = fmul float %0, %2
  %mul4.i = fmul float %1, %3
  %sub.i = fsub float %mul.i, %mul4.i
  %mul5.i = fmul float %1, %2
  %mul6.i = fmul float %0, %3
  %add.i4 = fadd float %mul5.i, %mul6.i
  %4 = bitcast i32 %ldd.sroa.0.0 to float
  %add.i = fadd float %sub.i, %4
  %5 = bitcast float %add.i to i32
  %6 = bitcast i32 %ldd.sroa.6.0 to float
  %add4.i = fadd float %add.i4, %6
  %7 = bitcast float %add4.i to i32
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  store i32 %ldd.sroa.0.0, i32* bitcast (%"struct.std::complex"* @dd to i32*), align 4
  store i32 %ldd.sroa.6.0, i32* bitcast (float* getelementptr inbounds (%"struct.std::complex", %"struct.std::complex"* @dd, i64 0, i32 0, i32 1) to i32*), align 4
  ret void

; CHECK: phi float
; CHECK: store float
; CHECK-NOT: bitcast
}


define void @multi_phi(i32 signext %n) {
entry:
  br label %for.cond

for.cond:
  %ldd.sroa.0.0 = phi i32 [ 0, %entry ], [ %9, %odd.bb ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %odd.bb ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %0 = load float, float* getelementptr inbounds (%"struct.std::complex", %"struct.std::complex"* @dd, i64 0, i32 0, i32 0), align 4
  %1 = load float, float* getelementptr inbounds (%"struct.std::complex", %"struct.std::complex"* @dd, i64 0, i32 0, i32 1), align 4
  %2 = load float, float* getelementptr inbounds (%"struct.std::complex", %"struct.std::complex"* @dd2, i64 0, i32 0, i32 0), align 4
  %3 = load float, float* getelementptr inbounds (%"struct.std::complex", %"struct.std::complex"* @dd2, i64 0, i32 0, i32 1), align 4
  %mul.i = fmul float %0, %2
  %mul4.i = fmul float %1, %3
  %sub.i = fsub float %mul.i, %mul4.i
  %4 = bitcast i32 %ldd.sroa.0.0 to float
  %add.i = fadd float %sub.i, %4
  %5 = bitcast float %add.i to i32
  %inc = add nsw i32 %i.0, 1
  %bit0 = and i32 %inc, 1
  %even = icmp slt i32 %bit0, 1
  br i1 %even, label %even.bb, label %odd.bb

even.bb:
  %6 = bitcast i32 %5 to float
  %7 = fadd float %sub.i, %6
  %8 = bitcast float %7 to i32
  br label %odd.bb

odd.bb:
  %9 = phi i32 [ %5, %for.body ], [ %8, %even.bb ]
  br label %for.cond

for.end:
  store i32 %ldd.sroa.0.0, i32* bitcast (%"struct.std::complex"* @dd to i32*), align 4
  ret void

; CHECK-LABEL: @multi_phi(
; CHECK: phi float
; CHECK: store float
; CHECK-NOT: bitcast
}
