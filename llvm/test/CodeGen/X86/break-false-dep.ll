; RUN: llc < %s -mtriple=x86_64-linux -mattr=+sse2 -mcpu=nehalem | FileCheck %s --check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-win32 -mattr=+sse2 -mcpu=nehalem | FileCheck %s --check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-win32 -mattr=+avx -mcpu=corei7-avx | FileCheck %s --check-prefix=AVX

define double @t1(float* nocapture %x) nounwind readonly ssp {
entry:
; SSE-LABEL: t1:
; SSE: movss ([[A0:%rdi|%rcx]]), %xmm0
; SSE: cvtss2sd %xmm0, %xmm0

  %0 = load float, float* %x, align 4
  %1 = fpext float %0 to double
  ret double %1
}

define float @t2(double* nocapture %x) nounwind readonly ssp optsize {
entry:
; SSE-LABEL: t2:
; SSE: cvtsd2ss ([[A0]]), %xmm0
  %0 = load double, double* %x, align 8
  %1 = fptrunc double %0 to float
  ret float %1
}

define float @squirtf(float* %x) nounwind {
entry:
; SSE-LABEL: squirtf:
; SSE: movss ([[A0]]), %xmm0
; SSE: sqrtss %xmm0, %xmm0
  %z = load float, float* %x
  %t = call float @llvm.sqrt.f32(float %z)
  ret float %t
}

define double @squirt(double* %x) nounwind {
entry:
; SSE-LABEL: squirt:
; SSE: movsd ([[A0]]), %xmm0
; SSE: sqrtsd %xmm0, %xmm0
  %z = load double, double* %x
  %t = call double @llvm.sqrt.f64(double %z)
  ret double %t
}

define float @squirtf_size(float* %x) nounwind optsize {
entry:
; SSE-LABEL: squirtf_size:
; SSE: sqrtss ([[A0]]), %xmm0
  %z = load float, float* %x
  %t = call float @llvm.sqrt.f32(float %z)
  ret float %t
}

define double @squirt_size(double* %x) nounwind optsize {
entry:
; SSE-LABEL: squirt_size:
; SSE: sqrtsd ([[A0]]), %xmm0
  %z = load double, double* %x
  %t = call double @llvm.sqrt.f64(double %z)
  ret double %t
}

declare float @llvm.sqrt.f32(float)
declare double @llvm.sqrt.f64(double)

; SSE-LABEL: loopdep1
; SSE: for.body{{$}}
;
; This loop contains two cvtsi2ss instructions that update the same xmm
; register.  Verify that the execution dependency fix pass breaks those
; dependencies by inserting xorps instructions.
;
; If the register allocator chooses different registers for the two cvtsi2ss
; instructions, they are still dependent on themselves.
; SSE: xorps [[XMM1:%xmm[0-9]+]]
; SSE: , [[XMM1]]
; SSE: cvtsi2ssl %{{.*}}, [[XMM1]]
; SSE: xorps [[XMM2:%xmm[0-9]+]]
; SSE: , [[XMM2]]
; SSE: cvtsi2ssl %{{.*}}, [[XMM2]]
;
define float @loopdep1(i32 %m) nounwind uwtable readnone ssp {
entry:
  %tobool3 = icmp eq i32 %m, 0
  br i1 %tobool3, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %m.addr.07 = phi i32 [ %dec, %for.body ], [ %m, %entry ]
  %s1.06 = phi float [ %add, %for.body ], [ 0.000000e+00, %entry ]
  %s2.05 = phi float [ %add2, %for.body ], [ 0.000000e+00, %entry ]
  %n.04 = phi i32 [ %inc, %for.body ], [ 1, %entry ]
  %conv = sitofp i32 %n.04 to float
  %add = fadd float %s1.06, %conv
  %conv1 = sitofp i32 %m.addr.07 to float
  %add2 = fadd float %s2.05, %conv1
  %inc = add nsw i32 %n.04, 1
  %dec = add nsw i32 %m.addr.07, -1
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %s1.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %s2.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add2, %for.body ]
  %sub = fsub float %s1.0.lcssa, %s2.0.lcssa
  ret float %sub
}

; rdar:15221834 False AVX register dependencies cause 5x slowdown on
; flops-6. Make sure the unused register read by vcvtsi2sdq is zeroed
; to avoid cyclic dependence on a write to the same register in a
; previous iteration.

; AVX-LABEL: loopdep2:
; AVX-LABEL: %loop
; AVX: vxorps %[[REG:xmm.]], %{{xmm.}}, %{{xmm.}}
; AVX: vcvtsi2sdq %{{r[0-9a-x]+}}, %[[REG]], %{{xmm.}}
; SSE-LABEL: loopdep2:
; SSE-LABEL: %loop
; SSE: xorps %[[REG:xmm.]], %[[REG]]
; SSE: cvtsi2sdq %{{r[0-9a-x]+}}, %[[REG]]
define i64 @loopdep2(i64* nocapture %x, double* nocapture %y) nounwind {
entry:
  %vx = load i64, i64* %x
  br label %loop
loop:
  %i = phi i64 [ 1, %entry ], [ %inc, %loop ]
  %s1 = phi i64 [ %vx, %entry ], [ %s2, %loop ]
  %fi = sitofp i64 %i to double
  %vy = load double, double* %y
  %fipy = fadd double %fi, %vy
  %iipy = fptosi double %fipy to i64
  %s2 = add i64 %s1, %iipy
  %inc = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %inc, 156250000
  br i1 %exitcond, label %ret, label %loop
ret:
  ret i64 %s2
}

; This loop contains a cvtsi2sd instruction that has a loop-carried
; false dependency on an xmm that is modified by other scalar instructions
; that follow it in the loop. Additionally, the source of convert is a
; memory operand. Verify the execution dependency fix pass breaks this
; dependency by inserting a xor before the convert.
@x = common global [1024 x double] zeroinitializer, align 16
@y = common global [1024 x double] zeroinitializer, align 16
@z = common global [1024 x double] zeroinitializer, align 16
@w = common global [1024 x double] zeroinitializer, align 16
@v = common global [1024 x i32] zeroinitializer, align 16

define void @loopdep3() {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc14, %entry
  %i.025 = phi i32 [ 0, %entry ], [ %inc15, %for.inc14 ]
  br label %for.body3

for.body3:
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx = getelementptr inbounds [1024 x i32], [1024 x i32]* @v, i64 0, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %conv = sitofp i32 %0 to double
  %arrayidx5 = getelementptr inbounds [1024 x double], [1024 x double]* @x, i64 0, i64 %indvars.iv
  %1 = load double, double* %arrayidx5, align 8
  %mul = fmul double %conv, %1
  %arrayidx7 = getelementptr inbounds [1024 x double], [1024 x double]* @y, i64 0, i64 %indvars.iv
  %2 = load double, double* %arrayidx7, align 8
  %mul8 = fmul double %mul, %2
  %arrayidx10 = getelementptr inbounds [1024 x double], [1024 x double]* @z, i64 0, i64 %indvars.iv
  %3 = load double, double* %arrayidx10, align 8
  %mul11 = fmul double %mul8, %3
  %arrayidx13 = getelementptr inbounds [1024 x double], [1024 x double]* @w, i64 0, i64 %indvars.iv
  store double %mul11, double* %arrayidx13, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.inc14, label %for.body3

for.inc14:                                        ; preds = %for.body3
  %inc15 = add nsw i32 %i.025, 1
  %exitcond26 = icmp eq i32 %inc15, 100000
  br i1 %exitcond26, label %for.end16, label %for.cond1.preheader

for.end16:                                        ; preds = %for.inc14
  ret void

;SSE-LABEL:@loopdep3
;SSE: xorps [[XMM0:%xmm[0-9]+]], [[XMM0]]
;SSE-NEXT: cvtsi2sdl {{.*}}, [[XMM0]]
;SSE-NEXT: mulsd {{.*}}, [[XMM0]]
;SSE-NEXT: mulsd {{.*}}, [[XMM0]]
;SSE-NEXT: mulsd {{.*}}, [[XMM0]]
;SSE-NEXT: movsd [[XMM0]],
;AVX-LABEL:@loopdep3
;AVX: vxorps [[XMM0:%xmm[0-9]+]], [[XMM0]]
;AVX-NEXT: vcvtsi2sdl {{.*}}, [[XMM0]], [[XMM0]]
;AVX-NEXT: vmulsd {{.*}}, [[XMM0]], [[XMM0]]
;AVX-NEXT: vmulsd {{.*}}, [[XMM0]], [[XMM0]]
;AVX-NEXT: vmulsd {{.*}}, [[XMM0]], [[XMM0]]
;AVX-NEXT: vmovsd [[XMM0]],
}
