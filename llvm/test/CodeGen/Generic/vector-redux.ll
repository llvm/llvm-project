; RUN: llc < %s -debug-only=isel -o /dev/null 2>&1 | FileCheck %s
; REQUIRES: asserts

@a = global [1024 x i32] zeroinitializer, align 16

define i32 @reduce_add() {
; CHECK-LABEL: reduce_add
; CHECK:       Detected a reduction operation: {{.*}} add
; CHECK:       Detected a reduction operation: {{.*}} add
; CHECK:       Detected a reduction operation: {{.*}} add
; CHECK:       Detected a reduction operation: {{.*}} add
; CHECK:       Detected a reduction operation: {{.*}} add
; CHECK:       Detected a reduction operation: {{.*}} add
; CHECK:       Detected a reduction operation: {{.*}} add
; CHECK:       Detected a reduction operation: {{.*}} add
; CHECK:       Detected a reduction operation: {{.*}} add
; CHECK:       Detected a reduction operation: {{.*}} add
; CHECK:       Detected a reduction operation: {{.*}} add

min.iters.checked:
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %min.iters.checked ], [ %index.next.4, %vector.body ]
  %vec.phi = phi <4 x i32> [ zeroinitializer, %min.iters.checked ], [ %28, %vector.body ]
  %vec.phi4 = phi <4 x i32> [ zeroinitializer, %min.iters.checked ], [ %29, %vector.body ]
  %0 = getelementptr inbounds [1024 x i32], [1024 x i32]* @a, i64 0, i64 %index
  %1 = bitcast i32* %0 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %1, align 16
  %2 = getelementptr i32, i32* %0, i64 4
  %3 = bitcast i32* %2 to <4 x i32>*
  %wide.load5 = load <4 x i32>, <4 x i32>* %3, align 16
  %4 = add nsw <4 x i32> %wide.load, %vec.phi
  %5 = add nsw <4 x i32> %wide.load5, %vec.phi4
  %index.next = add nuw nsw i64 %index, 8
  %6 = getelementptr inbounds [1024 x i32], [1024 x i32]* @a, i64 0, i64 %index.next
  %7 = bitcast i32* %6 to <4 x i32>*
  %wide.load.1 = load <4 x i32>, <4 x i32>* %7, align 16
  %8 = getelementptr i32, i32* %6, i64 4
  %9 = bitcast i32* %8 to <4 x i32>*
  %wide.load5.1 = load <4 x i32>, <4 x i32>* %9, align 16
  %10 = add nsw <4 x i32> %wide.load.1, %4
  %11 = add nsw <4 x i32> %wide.load5.1, %5
  %index.next.1 = add nsw i64 %index, 16
  %12 = getelementptr inbounds [1024 x i32], [1024 x i32]* @a, i64 0, i64 %index.next.1
  %13 = bitcast i32* %12 to <4 x i32>*
  %wide.load.2 = load <4 x i32>, <4 x i32>* %13, align 16
  %14 = getelementptr i32, i32* %12, i64 4
  %15 = bitcast i32* %14 to <4 x i32>*
  %wide.load5.2 = load <4 x i32>, <4 x i32>* %15, align 16
  %16 = add nsw <4 x i32> %wide.load.2, %10
  %17 = add nsw <4 x i32> %wide.load5.2, %11
  %index.next.2 = add nsw i64 %index, 24
  %18 = getelementptr inbounds [1024 x i32], [1024 x i32]* @a, i64 0, i64 %index.next.2
  %19 = bitcast i32* %18 to <4 x i32>*
  %wide.load.3 = load <4 x i32>, <4 x i32>* %19, align 16
  %20 = getelementptr i32, i32* %18, i64 4
  %21 = bitcast i32* %20 to <4 x i32>*
  %wide.load5.3 = load <4 x i32>, <4 x i32>* %21, align 16
  %22 = add nsw <4 x i32> %wide.load.3, %16
  %23 = add nsw <4 x i32> %wide.load5.3, %17
  %index.next.3 = add nsw i64 %index, 32
  %24 = getelementptr inbounds [1024 x i32], [1024 x i32]* @a, i64 0, i64 %index.next.3
  %25 = bitcast i32* %24 to <4 x i32>*
  %wide.load.4 = load <4 x i32>, <4 x i32>* %25, align 16
  %26 = getelementptr i32, i32* %24, i64 4
  %27 = bitcast i32* %26 to <4 x i32>*
  %wide.load5.4 = load <4 x i32>, <4 x i32>* %27, align 16
  %28 = add nsw <4 x i32> %wide.load.4, %22
  %29 = add nsw <4 x i32> %wide.load5.4, %23
  %index.next.4 = add nsw i64 %index, 40
  %30 = icmp eq i64 %index.next.4, 1000
  br i1 %30, label %middle.block, label %vector.body

middle.block:
  %.lcssa10 = phi <4 x i32> [ %29, %vector.body ]
  %.lcssa = phi <4 x i32> [ %28, %vector.body ]
  %bin.rdx = add <4 x i32> %.lcssa10, %.lcssa
  %rdx.shuf = shufflevector <4 x i32> %bin.rdx, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx6 = add <4 x i32> %bin.rdx, %rdx.shuf
  %rdx.shuf7 = shufflevector <4 x i32> %bin.rdx6, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = add <4 x i32> %bin.rdx6, %rdx.shuf7
  %31 = extractelement <4 x i32> %bin.rdx8, i32 0
  ret i32 %31
}

define i32 @reduce_and() {
; CHECK-LABEL: reduce_and
; CHECK:       Detected a reduction operation: {{.*}} and
; CHECK:       Detected a reduction operation: {{.*}} and
; CHECK:       Detected a reduction operation: {{.*}} and
; CHECK:       Detected a reduction operation: {{.*}} and
; CHECK:       Detected a reduction operation: {{.*}} and
; CHECK:       Detected a reduction operation: {{.*}} and
; CHECK:       Detected a reduction operation: {{.*}} and
; CHECK:       Detected a reduction operation: {{.*}} and
; CHECK:       Detected a reduction operation: {{.*}} and

entry:
  br label %vector.body

vector.body:
  %lsr.iv = phi i64 [ %lsr.iv.next, %vector.body ], [ -4096, %entry ]
  %vec.phi = phi <4 x i32> [ <i32 -1, i32 -1, i32 -1, i32 -1>, %entry ], [ %6, %vector.body ]
  %vec.phi9 = phi <4 x i32> [ <i32 -1, i32 -1, i32 -1, i32 -1>, %entry ], [ %7, %vector.body ]
  %uglygep33 = getelementptr i8, i8* bitcast ([1024 x i32]* @a to i8*), i64 %lsr.iv
  %uglygep3334 = bitcast i8* %uglygep33 to <4 x i32>*
  %scevgep35 = getelementptr <4 x i32>, <4 x i32>* %uglygep3334, i64 256
  %wide.load = load <4 x i32>, <4 x i32>* %scevgep35, align 16
  %scevgep36 = getelementptr <4 x i32>, <4 x i32>* %uglygep3334, i64 257
  %wide.load10 = load <4 x i32>, <4 x i32>* %scevgep36, align 16
  %0 = and <4 x i32> %wide.load, %vec.phi
  %1 = and <4 x i32> %wide.load10, %vec.phi9
  %uglygep30 = getelementptr i8, i8* bitcast ([1024 x i32]* @a to i8*), i64 %lsr.iv
  %uglygep3031 = bitcast i8* %uglygep30 to <4 x i32>*
  %scevgep32 = getelementptr <4 x i32>, <4 x i32>* %uglygep3031, i64 258
  %wide.load.1 = load <4 x i32>, <4 x i32>* %scevgep32, align 16
  %uglygep27 = getelementptr i8, i8* bitcast ([1024 x i32]* @a to i8*), i64 %lsr.iv
  %uglygep2728 = bitcast i8* %uglygep27 to <4 x i32>*
  %scevgep29 = getelementptr <4 x i32>, <4 x i32>* %uglygep2728, i64 259
  %wide.load10.1 = load <4 x i32>, <4 x i32>* %scevgep29, align 16
  %2 = and <4 x i32> %wide.load.1, %0
  %3 = and <4 x i32> %wide.load10.1, %1
  %uglygep24 = getelementptr i8, i8* bitcast ([1024 x i32]* @a to i8*), i64 %lsr.iv
  %uglygep2425 = bitcast i8* %uglygep24 to <4 x i32>*
  %scevgep26 = getelementptr <4 x i32>, <4 x i32>* %uglygep2425, i64 260
  %wide.load.2 = load <4 x i32>, <4 x i32>* %scevgep26, align 16
  %uglygep21 = getelementptr i8, i8* bitcast ([1024 x i32]* @a to i8*), i64 %lsr.iv
  %uglygep2122 = bitcast i8* %uglygep21 to <4 x i32>*
  %scevgep23 = getelementptr <4 x i32>, <4 x i32>* %uglygep2122, i64 261
  %wide.load10.2 = load <4 x i32>, <4 x i32>* %scevgep23, align 16
  %4 = and <4 x i32> %wide.load.2, %2
  %5 = and <4 x i32> %wide.load10.2, %3
  %uglygep18 = getelementptr i8, i8* bitcast ([1024 x i32]* @a to i8*), i64 %lsr.iv
  %uglygep1819 = bitcast i8* %uglygep18 to <4 x i32>*
  %scevgep20 = getelementptr <4 x i32>, <4 x i32>* %uglygep1819, i64 262
  %wide.load.3 = load <4 x i32>, <4 x i32>* %scevgep20, align 16
  %uglygep = getelementptr i8, i8* bitcast ([1024 x i32]* @a to i8*), i64 %lsr.iv
  %uglygep17 = bitcast i8* %uglygep to <4 x i32>*
  %scevgep = getelementptr <4 x i32>, <4 x i32>* %uglygep17, i64 263
  %wide.load10.3 = load <4 x i32>, <4 x i32>* %scevgep, align 16
  %6 = and <4 x i32> %wide.load.3, %4
  %7 = and <4 x i32> %wide.load10.3, %5
  %lsr.iv.next = add nsw i64 %lsr.iv, 128
  %8 = icmp eq i64 %lsr.iv.next, 0
  br i1 %8, label %middle.block, label %vector.body

middle.block:
  %bin.rdx = and <4 x i32> %7, %6
  %rdx.shuf = shufflevector <4 x i32> %bin.rdx, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx11 = and <4 x i32> %bin.rdx, %rdx.shuf
  %rdx.shuf12 = shufflevector <4 x i32> %bin.rdx11, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx13 = and <4 x i32> %bin.rdx11, %rdx.shuf12
  %9 = extractelement <4 x i32> %bin.rdx13, i32 0
  ret i32 %9
}

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
