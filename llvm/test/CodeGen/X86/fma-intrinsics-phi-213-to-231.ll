; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2,+fma | FileCheck %s

; CHECK-LABEL: fmaddsubpd_loop_128:
; CHECK:   vfmaddsub231pd %xmm1, %xmm0, %xmm2
; CHECK:   vmovaps %xmm2, %xmm0
; CHECK-NEXT: retq
define <2 x double> @fmaddsubpd_loop_128(i32 %iter, <2 x double> %a, <2 x double> %b, <2 x double> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <2 x double> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <2 x double> @llvm.x86.fma.vfmaddsub.pd(<2 x double> %a, <2 x double> %b, <2 x double> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <2 x double> %c.addr.0
}

; CHECK-LABEL: fmsubaddpd_loop_128:
; CHECK:   vfmsubadd231pd %xmm1, %xmm0, %xmm2
; CHECK:   vmovaps %xmm2, %xmm0
; CHECK-NEXT: retq
define <2 x double> @fmsubaddpd_loop_128(i32 %iter, <2 x double> %a, <2 x double> %b, <2 x double> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <2 x double> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <2 x double> @llvm.x86.fma.vfmsubadd.pd(<2 x double> %a, <2 x double> %b, <2 x double> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <2 x double> %c.addr.0
}

; CHECK-LABEL: fmaddpd_loop_128:
; CHECK:   vfmadd231pd %xmm1, %xmm0, %xmm2
; CHECK:   vmovaps %xmm2, %xmm0
; CHECK-NEXT: retq
define <2 x double> @fmaddpd_loop_128(i32 %iter, <2 x double> %a, <2 x double> %b, <2 x double> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <2 x double> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <2 x double> @llvm.x86.fma.vfmadd.pd(<2 x double> %a, <2 x double> %b, <2 x double> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <2 x double> %c.addr.0
}

; CHECK-LABEL: fmsubpd_loop_128:
; CHECK:   vfmsub231pd %xmm1, %xmm0, %xmm2
; CHECK:   vmovaps %xmm2, %xmm0
; CHECK-NEXT: retq
define <2 x double> @fmsubpd_loop_128(i32 %iter, <2 x double> %a, <2 x double> %b, <2 x double> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <2 x double> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <2 x double> @llvm.x86.fma.vfmsub.pd(<2 x double> %a, <2 x double> %b, <2 x double> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <2 x double> %c.addr.0
}

; CHECK-LABEL: fnmaddpd_loop_128:
; CHECK:   vfnmadd231pd %xmm1, %xmm0, %xmm2
; CHECK:   vmovaps %xmm2, %xmm0
; CHECK-NEXT: retq
define <2 x double> @fnmaddpd_loop_128(i32 %iter, <2 x double> %a, <2 x double> %b, <2 x double> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <2 x double> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <2 x double> @llvm.x86.fma.vfnmadd.pd(<2 x double> %a, <2 x double> %b, <2 x double> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <2 x double> %c.addr.0
}

; CHECK-LABEL: fnmsubpd_loop_128:
; CHECK:   vfnmsub231pd %xmm1, %xmm0, %xmm2
; CHECK:   vmovaps %xmm2, %xmm0
; CHECK-NEXT: retq
define <2 x double> @fnmsubpd_loop_128(i32 %iter, <2 x double> %a, <2 x double> %b, <2 x double> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <2 x double> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <2 x double> @llvm.x86.fma.vfnmsub.pd(<2 x double> %a, <2 x double> %b, <2 x double> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <2 x double> %c.addr.0
}

declare <2 x double> @llvm.x86.fma.vfmaddsub.pd(<2 x double>, <2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.fma.vfmsubadd.pd(<2 x double>, <2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.fma.vfmadd.pd(<2 x double>, <2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.fma.vfmsub.pd(<2 x double>, <2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.fma.vfnmadd.pd(<2 x double>, <2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.fma.vfnmsub.pd(<2 x double>, <2 x double>, <2 x double>)


; CHECK-LABEL: fmaddsubps_loop_128:
; CHECK:   vfmaddsub231ps %xmm1, %xmm0, %xmm2
; CHECK:   vmovaps %xmm2, %xmm0
; CHECK-NEXT: retq
define <4 x float> @fmaddsubps_loop_128(i32 %iter, <4 x float> %a, <4 x float> %b, <4 x float> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <4 x float> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <4 x float> @llvm.x86.fma.vfmaddsub.ps(<4 x float> %a, <4 x float> %b, <4 x float> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <4 x float> %c.addr.0
}

; CHECK-LABEL: fmsubaddps_loop_128:
; CHECK:   vfmsubadd231ps %xmm1, %xmm0, %xmm2
; CHECK:   vmovaps %xmm2, %xmm0
; CHECK-NEXT: retq
define <4 x float> @fmsubaddps_loop_128(i32 %iter, <4 x float> %a, <4 x float> %b, <4 x float> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <4 x float> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <4 x float> @llvm.x86.fma.vfmsubadd.ps(<4 x float> %a, <4 x float> %b, <4 x float> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <4 x float> %c.addr.0
}

; CHECK-LABEL: fmaddps_loop_128:
; CHECK:   vfmadd231ps %xmm1, %xmm0, %xmm2
; CHECK:   vmovaps %xmm2, %xmm0
; CHECK-NEXT: retq
define <4 x float> @fmaddps_loop_128(i32 %iter, <4 x float> %a, <4 x float> %b, <4 x float> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <4 x float> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <4 x float> @llvm.x86.fma.vfmadd.ps(<4 x float> %a, <4 x float> %b, <4 x float> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <4 x float> %c.addr.0
}

; CHECK-LABEL: fmsubps_loop_128:
; CHECK:   vfmsub231ps %xmm1, %xmm0, %xmm2
; CHECK:   vmovaps %xmm2, %xmm0
; CHECK-NEXT: retq
define <4 x float> @fmsubps_loop_128(i32 %iter, <4 x float> %a, <4 x float> %b, <4 x float> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <4 x float> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <4 x float> @llvm.x86.fma.vfmsub.ps(<4 x float> %a, <4 x float> %b, <4 x float> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <4 x float> %c.addr.0
}

; CHECK-LABEL: fnmaddps_loop_128:
; CHECK:   vfnmadd231ps %xmm1, %xmm0, %xmm2
; CHECK:   vmovaps %xmm2, %xmm0
; CHECK-NEXT: retq
define <4 x float> @fnmaddps_loop_128(i32 %iter, <4 x float> %a, <4 x float> %b, <4 x float> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <4 x float> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <4 x float> @llvm.x86.fma.vfnmadd.ps(<4 x float> %a, <4 x float> %b, <4 x float> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <4 x float> %c.addr.0
}

; CHECK-LABEL: fnmsubps_loop_128:
; CHECK:   vfnmsub231ps %xmm1, %xmm0, %xmm2
; CHECK:   vmovaps %xmm2, %xmm0
; CHECK-NEXT: retq
define <4 x float> @fnmsubps_loop_128(i32 %iter, <4 x float> %a, <4 x float> %b, <4 x float> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <4 x float> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <4 x float> @llvm.x86.fma.vfnmsub.ps(<4 x float> %a, <4 x float> %b, <4 x float> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <4 x float> %c.addr.0
}

declare <4 x float> @llvm.x86.fma.vfmaddsub.ps(<4 x float>, <4 x float>, <4 x float>)
declare <4 x float> @llvm.x86.fma.vfmsubadd.ps(<4 x float>, <4 x float>, <4 x float>)
declare <4 x float> @llvm.x86.fma.vfmadd.ps(<4 x float>, <4 x float>, <4 x float>)
declare <4 x float> @llvm.x86.fma.vfmsub.ps(<4 x float>, <4 x float>, <4 x float>)
declare <4 x float> @llvm.x86.fma.vfnmadd.ps(<4 x float>, <4 x float>, <4 x float>)
declare <4 x float> @llvm.x86.fma.vfnmsub.ps(<4 x float>, <4 x float>, <4 x float>)

; CHECK-LABEL: fmaddsubpd_loop_256:
; CHECK:   vfmaddsub231pd %ymm1, %ymm0, %ymm2
; CHECK:   vmovaps %ymm2, %ymm0
; CHECK-NEXT: retq
define <4 x double> @fmaddsubpd_loop_256(i32 %iter, <4 x double> %a, <4 x double> %b, <4 x double> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <4 x double> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double> %a, <4 x double> %b, <4 x double> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <4 x double> %c.addr.0
}

; CHECK-LABEL: fmsubaddpd_loop_256:
; CHECK:   vfmsubadd231pd %ymm1, %ymm0, %ymm2
; CHECK:   vmovaps %ymm2, %ymm0
; CHECK-NEXT: retq
define <4 x double> @fmsubaddpd_loop_256(i32 %iter, <4 x double> %a, <4 x double> %b, <4 x double> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <4 x double> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <4 x double> @llvm.x86.fma.vfmsubadd.pd.256(<4 x double> %a, <4 x double> %b, <4 x double> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <4 x double> %c.addr.0
}

; CHECK-LABEL: fmaddpd_loop_256:
; CHECK:   vfmadd231pd %ymm1, %ymm0, %ymm2
; CHECK:   vmovaps %ymm2, %ymm0
; CHECK-NEXT: retq
define <4 x double> @fmaddpd_loop_256(i32 %iter, <4 x double> %a, <4 x double> %b, <4 x double> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <4 x double> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <4 x double> @llvm.x86.fma.vfmadd.pd.256(<4 x double> %a, <4 x double> %b, <4 x double> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <4 x double> %c.addr.0
}

; CHECK-LABEL: fmsubpd_loop_256:
; CHECK:   vfmsub231pd %ymm1, %ymm0, %ymm2
; CHECK:   vmovaps %ymm2, %ymm0
; CHECK-NEXT: retq
define <4 x double> @fmsubpd_loop_256(i32 %iter, <4 x double> %a, <4 x double> %b, <4 x double> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <4 x double> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <4 x double> @llvm.x86.fma.vfmsub.pd.256(<4 x double> %a, <4 x double> %b, <4 x double> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <4 x double> %c.addr.0
}

; CHECK-LABEL: fnmaddpd_loop_256:
; CHECK:   vfnmadd231pd %ymm1, %ymm0, %ymm2
; CHECK:   vmovaps %ymm2, %ymm0
; CHECK-NEXT: retq
define <4 x double> @fnmaddpd_loop_256(i32 %iter, <4 x double> %a, <4 x double> %b, <4 x double> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <4 x double> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <4 x double> @llvm.x86.fma.vfnmadd.pd.256(<4 x double> %a, <4 x double> %b, <4 x double> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <4 x double> %c.addr.0
}

; CHECK-LABEL: fnmsubpd_loop_256:
; CHECK:   vfnmsub231pd %ymm1, %ymm0, %ymm2
; CHECK:   vmovaps %ymm2, %ymm0
; CHECK-NEXT: retq
define <4 x double> @fnmsubpd_loop_256(i32 %iter, <4 x double> %a, <4 x double> %b, <4 x double> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <4 x double> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <4 x double> @llvm.x86.fma.vfnmsub.pd.256(<4 x double> %a, <4 x double> %b, <4 x double> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <4 x double> %c.addr.0
}

declare <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double>, <4 x double>, <4 x double>)
declare <4 x double> @llvm.x86.fma.vfmsubadd.pd.256(<4 x double>, <4 x double>, <4 x double>)
declare <4 x double> @llvm.x86.fma.vfmadd.pd.256(<4 x double>, <4 x double>, <4 x double>)
declare <4 x double> @llvm.x86.fma.vfmsub.pd.256(<4 x double>, <4 x double>, <4 x double>)
declare <4 x double> @llvm.x86.fma.vfnmadd.pd.256(<4 x double>, <4 x double>, <4 x double>)
declare <4 x double> @llvm.x86.fma.vfnmsub.pd.256(<4 x double>, <4 x double>, <4 x double>)


; CHECK-LABEL: fmaddsubps_loop_256:
; CHECK:   vfmaddsub231ps %ymm1, %ymm0, %ymm2
; CHECK:   vmovaps %ymm2, %ymm0
; CHECK-NEXT: retq
define <8 x float> @fmaddsubps_loop_256(i32 %iter, <8 x float> %a, <8 x float> %b, <8 x float> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <8 x float> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <8 x float> @llvm.x86.fma.vfmaddsub.ps.256(<8 x float> %a, <8 x float> %b, <8 x float> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <8 x float> %c.addr.0
}

; CHECK-LABEL: fmsubaddps_loop_256:
; CHECK:   vfmsubadd231ps %ymm1, %ymm0, %ymm2
; CHECK:   vmovaps %ymm2, %ymm0
; CHECK-NEXT: retq
define <8 x float> @fmsubaddps_loop_256(i32 %iter, <8 x float> %a, <8 x float> %b, <8 x float> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <8 x float> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <8 x float> @llvm.x86.fma.vfmsubadd.ps.256(<8 x float> %a, <8 x float> %b, <8 x float> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <8 x float> %c.addr.0
}

; CHECK-LABEL: fmaddps_loop_256:
; CHECK:   vfmadd231ps %ymm1, %ymm0, %ymm2
; CHECK:   vmovaps %ymm2, %ymm0
; CHECK-NEXT: retq
define <8 x float> @fmaddps_loop_256(i32 %iter, <8 x float> %a, <8 x float> %b, <8 x float> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <8 x float> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <8 x float> @llvm.x86.fma.vfmadd.ps.256(<8 x float> %a, <8 x float> %b, <8 x float> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <8 x float> %c.addr.0
}

; CHECK-LABEL: fmsubps_loop_256:
; CHECK:   vfmsub231ps %ymm1, %ymm0, %ymm2
; CHECK:   vmovaps %ymm2, %ymm0
; CHECK-NEXT: retq
define <8 x float> @fmsubps_loop_256(i32 %iter, <8 x float> %a, <8 x float> %b, <8 x float> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <8 x float> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <8 x float> @llvm.x86.fma.vfmsub.ps.256(<8 x float> %a, <8 x float> %b, <8 x float> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <8 x float> %c.addr.0
}

; CHECK-LABEL: fnmaddps_loop_256:
; CHECK:   vfnmadd231ps %ymm1, %ymm0, %ymm2
; CHECK:   vmovaps %ymm2, %ymm0
; CHECK-NEXT: retq
define <8 x float> @fnmaddps_loop_256(i32 %iter, <8 x float> %a, <8 x float> %b, <8 x float> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <8 x float> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <8 x float> @llvm.x86.fma.vfnmadd.ps.256(<8 x float> %a, <8 x float> %b, <8 x float> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <8 x float> %c.addr.0
}

; CHECK-LABEL: fnmsubps_loop_256:
; CHECK:   vfnmsub231ps %ymm1, %ymm0, %ymm2
; CHECK:   vmovaps %ymm2, %ymm0
; CHECK-NEXT: retq
define <8 x float> @fnmsubps_loop_256(i32 %iter, <8 x float> %a, <8 x float> %b, <8 x float> %c) {
entry:
  br label %for.cond

for.cond:
  %c.addr.0 = phi <8 x float> [ %c, %entry ], [ %0, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %iter
  br i1 %cmp, label %for.body, label %for.end

for.body:
  br label %for.inc

for.inc:
  %0 = call <8 x float> @llvm.x86.fma.vfnmsub.ps.256(<8 x float> %a, <8 x float> %b, <8 x float> %c.addr.0)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret <8 x float> %c.addr.0
}

declare <8 x float> @llvm.x86.fma.vfmaddsub.ps.256(<8 x float>, <8 x float>, <8 x float>)
declare <8 x float> @llvm.x86.fma.vfmsubadd.ps.256(<8 x float>, <8 x float>, <8 x float>)
declare <8 x float> @llvm.x86.fma.vfmadd.ps.256(<8 x float>, <8 x float>, <8 x float>)
declare <8 x float> @llvm.x86.fma.vfmsub.ps.256(<8 x float>, <8 x float>, <8 x float>)
declare <8 x float> @llvm.x86.fma.vfnmadd.ps.256(<8 x float>, <8 x float>, <8 x float>)
declare <8 x float> @llvm.x86.fma.vfnmsub.ps.256(<8 x float>, <8 x float>, <8 x float>)
