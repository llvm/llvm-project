; RUN: llc < %s -debug-only=isel -o /dev/null 2>&1 | FileCheck %s
; REQUIRES: asserts

@a = global [1024 x i32] zeroinitializer, align 16

define float @reduce_add_float(float* nocapture readonly %a) {
; CHECK-LABEL: reduce_add_float
; CHECK:       Detected a reduction operation: {{.*}} fadd fast
; CHECK:       Detected a reduction operation: {{.*}} fadd fast
; CHECK:       Detected a reduction operation: {{.*}} fadd fast
; CHECK:       Detected a reduction operation: {{.*}} fadd fast
; CHECK:       Detected a reduction operation: {{.*}} fadd fast
; CHECK:       Detected a reduction operation: {{.*}} fadd fast
; CHECK:       Detected a reduction operation: {{.*}} fadd fast
; CHECK:       Detected a reduction operation: {{.*}} fadd fast
; CHECK:       Detected a reduction operation: {{.*}} fadd fast
; CHECK:       Detected a reduction operation: {{.*}} fadd fast
; CHECK:       Detected a reduction operation: {{.*}} fadd fast
;
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next.4, %vector.body ]
  %vec.phi = phi <4 x float> [ zeroinitializer, %entry ], [ %28, %vector.body ]
  %vec.phi9 = phi <4 x float> [ zeroinitializer, %entry ], [ %29, %vector.body ]
  %0 = getelementptr inbounds float, float* %a, i64 %index
  %1 = bitcast float* %0 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %1, align 4
  %2 = getelementptr float, float* %0, i64 4
  %3 = bitcast float* %2 to <4 x float>*
  %wide.load10 = load <4 x float>, <4 x float>* %3, align 4
  %4 = fadd fast <4 x float> %wide.load, %vec.phi
  %5 = fadd fast <4 x float> %wide.load10, %vec.phi9
  %index.next = add nuw nsw i64 %index, 8
  %6 = getelementptr inbounds float, float* %a, i64 %index.next
  %7 = bitcast float* %6 to <4 x float>*
  %wide.load.1 = load <4 x float>, <4 x float>* %7, align 4
  %8 = getelementptr float, float* %6, i64 4
  %9 = bitcast float* %8 to <4 x float>*
  %wide.load10.1 = load <4 x float>, <4 x float>* %9, align 4
  %10 = fadd fast <4 x float> %wide.load.1, %4
  %11 = fadd fast <4 x float> %wide.load10.1, %5
  %index.next.1 = add nsw i64 %index, 16
  %12 = getelementptr inbounds float, float* %a, i64 %index.next.1
  %13 = bitcast float* %12 to <4 x float>*
  %wide.load.2 = load <4 x float>, <4 x float>* %13, align 4
  %14 = getelementptr float, float* %12, i64 4
  %15 = bitcast float* %14 to <4 x float>*
  %wide.load10.2 = load <4 x float>, <4 x float>* %15, align 4
  %16 = fadd fast <4 x float> %wide.load.2, %10
  %17 = fadd fast <4 x float> %wide.load10.2, %11
  %index.next.2 = add nsw i64 %index, 24
  %18 = getelementptr inbounds float, float* %a, i64 %index.next.2
  %19 = bitcast float* %18 to <4 x float>*
  %wide.load.3 = load <4 x float>, <4 x float>* %19, align 4
  %20 = getelementptr float, float* %18, i64 4
  %21 = bitcast float* %20 to <4 x float>*
  %wide.load10.3 = load <4 x float>, <4 x float>* %21, align 4
  %22 = fadd fast <4 x float> %wide.load.3, %16
  %23 = fadd fast <4 x float> %wide.load10.3, %17
  %index.next.3 = add nsw i64 %index, 32
  %24 = getelementptr inbounds float, float* %a, i64 %index.next.3
  %25 = bitcast float* %24 to <4 x float>*
  %wide.load.4 = load <4 x float>, <4 x float>* %25, align 4
  %26 = getelementptr float, float* %24, i64 4
  %27 = bitcast float* %26 to <4 x float>*
  %wide.load10.4 = load <4 x float>, <4 x float>* %27, align 4
  %28 = fadd fast <4 x float> %wide.load.4, %22
  %29 = fadd fast <4 x float> %wide.load10.4, %23
  %index.next.4 = add nsw i64 %index, 40
  %30 = icmp eq i64 %index.next.4, 1000
  br i1 %30, label %middle.block, label %vector.body

middle.block:
  %.lcssa15 = phi <4 x float> [ %29, %vector.body ]
  %.lcssa = phi <4 x float> [ %28, %vector.body ]
  %bin.rdx = fadd fast <4 x float> %.lcssa15, %.lcssa
  %rdx.shuf = shufflevector <4 x float> %bin.rdx, <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx11 = fadd fast <4 x float> %bin.rdx, %rdx.shuf
  %rdx.shuf12 = shufflevector <4 x float> %bin.rdx11, <4 x float> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx13 = fadd fast <4 x float> %bin.rdx11, %rdx.shuf12
  %31 = extractelement <4 x float> %bin.rdx13, i32 0
  ret float %31
}
