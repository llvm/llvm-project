; RUN: llc < %s -march=x86 | not grep cmov

; LSR should be able to eliminate the max computations by
; making the loops use slt/ult comparisons instead of ne comparisons.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"

define void @foo(i8* %r, i32 %s, i32 %w, i32 %x, i8* %j, i32 %d) nounwind {
entry:
  %0 = mul i32 %x, %w
  %1 = mul i32 %x, %w
  %2 = sdiv i32 %1, 4
  %.sum2 = add i32 %2, %0
  %cond = icmp eq i32 %d, 1
  br i1 %cond, label %bb29, label %bb10.preheader

bb10.preheader:                                   ; preds = %entry
  %3 = icmp sgt i32 %x, 0
  br i1 %3, label %bb.nph9, label %bb18.loopexit

bb.nph7:                                          ; preds = %bb7.preheader
  %4 = mul i32 %y.08, %w
  %5 = mul i32 %y.08, %s
  %6 = add i32 %5, 1
  %tmp8 = icmp sgt i32 1, %w
  %smax9 = select i1 %tmp8, i32 1, i32 %w
  br label %bb6

bb6:                                              ; preds = %bb7, %bb.nph7
  %x.06 = phi i32 [ 0, %bb.nph7 ], [ %indvar.next7, %bb7 ]
  %7 = add i32 %x.06, %4
  %8 = shl i32 %x.06, 1
  %9 = add i32 %6, %8
  %10 = getelementptr i8, i8* %r, i32 %9
  %11 = load i8, i8* %10, align 1
  %12 = getelementptr i8, i8* %j, i32 %7
  store i8 %11, i8* %12, align 1
  br label %bb7

bb7:                                              ; preds = %bb6
  %indvar.next7 = add i32 %x.06, 1
  %exitcond10 = icmp ne i32 %indvar.next7, %smax9
  br i1 %exitcond10, label %bb6, label %bb7.bb9_crit_edge

bb7.bb9_crit_edge:                                ; preds = %bb7
  br label %bb9

bb9:                                              ; preds = %bb7.preheader, %bb7.bb9_crit_edge
  br label %bb10

bb10:                                             ; preds = %bb9
  %indvar.next11 = add i32 %y.08, 1
  %exitcond12 = icmp ne i32 %indvar.next11, %x
  br i1 %exitcond12, label %bb7.preheader, label %bb10.bb18.loopexit_crit_edge

bb10.bb18.loopexit_crit_edge:                     ; preds = %bb10
  br label %bb10.bb18.loopexit_crit_edge.split

bb10.bb18.loopexit_crit_edge.split:               ; preds = %bb.nph9, %bb10.bb18.loopexit_crit_edge
  br label %bb18.loopexit

bb.nph9:                                          ; preds = %bb10.preheader
  %13 = icmp sgt i32 %w, 0
  br i1 %13, label %bb.nph9.split, label %bb10.bb18.loopexit_crit_edge.split

bb.nph9.split:                                    ; preds = %bb.nph9
  br label %bb7.preheader

bb7.preheader:                                    ; preds = %bb.nph9.split, %bb10
  %y.08 = phi i32 [ 0, %bb.nph9.split ], [ %indvar.next11, %bb10 ]
  br i1 true, label %bb.nph7, label %bb9

bb.nph5:                                          ; preds = %bb18.loopexit
  %14 = sdiv i32 %w, 2
  %15 = icmp slt i32 %w, 2
  %16 = sdiv i32 %x, 2
  br i1 %15, label %bb18.bb20_crit_edge.split, label %bb.nph5.split

bb.nph5.split:                                    ; preds = %bb.nph5
  %tmp2 = icmp sgt i32 1, %16
  %smax3 = select i1 %tmp2, i32 1, i32 %16
  br label %bb13

bb13:                                             ; preds = %bb18, %bb.nph5.split
  %y.14 = phi i32 [ 0, %bb.nph5.split ], [ %indvar.next1, %bb18 ]
  %17 = mul i32 %14, %y.14
  %18 = shl i32 %y.14, 1
  %19 = srem i32 %y.14, 2
  %20 = add i32 %19, %18
  %21 = mul i32 %20, %s
  br i1 true, label %bb.nph3, label %bb17

bb.nph3:                                          ; preds = %bb13
  %22 = add i32 %17, %0
  %23 = add i32 %17, %.sum2
  %24 = sdiv i32 %w, 2
  %tmp = icmp sgt i32 1, %24
  %smax = select i1 %tmp, i32 1, i32 %24
  br label %bb14

bb14:                                             ; preds = %bb15, %bb.nph3
  %x.12 = phi i32 [ 0, %bb.nph3 ], [ %indvar.next, %bb15 ]
  %25 = shl i32 %x.12, 2
  %26 = add i32 %25, %21
  %27 = getelementptr i8, i8* %r, i32 %26
  %28 = load i8, i8* %27, align 1
  %.sum = add i32 %22, %x.12
  %29 = getelementptr i8, i8* %j, i32 %.sum
  store i8 %28, i8* %29, align 1
  %30 = shl i32 %x.12, 2
  %31 = or i32 %30, 2
  %32 = add i32 %31, %21
  %33 = getelementptr i8, i8* %r, i32 %32
  %34 = load i8, i8* %33, align 1
  %.sum6 = add i32 %23, %x.12
  %35 = getelementptr i8, i8* %j, i32 %.sum6
  store i8 %34, i8* %35, align 1
  br label %bb15

bb15:                                             ; preds = %bb14
  %indvar.next = add i32 %x.12, 1
  %exitcond = icmp ne i32 %indvar.next, %smax
  br i1 %exitcond, label %bb14, label %bb15.bb17_crit_edge

bb15.bb17_crit_edge:                              ; preds = %bb15
  br label %bb17

bb17:                                             ; preds = %bb15.bb17_crit_edge, %bb13
  br label %bb18

bb18.loopexit:                                    ; preds = %bb10.bb18.loopexit_crit_edge.split, %bb10.preheader
  %36 = icmp slt i32 %x, 2
  br i1 %36, label %bb20, label %bb.nph5

bb18:                                             ; preds = %bb17
  %indvar.next1 = add i32 %y.14, 1
  %exitcond4 = icmp ne i32 %indvar.next1, %smax3
  br i1 %exitcond4, label %bb13, label %bb18.bb20_crit_edge

bb18.bb20_crit_edge:                              ; preds = %bb18
  br label %bb18.bb20_crit_edge.split

bb18.bb20_crit_edge.split:                        ; preds = %bb18.bb20_crit_edge, %bb.nph5
  br label %bb20

bb20:                                             ; preds = %bb18.bb20_crit_edge.split, %bb18.loopexit
  switch i32 %d, label %return [
    i32 3, label %bb22
    i32 1, label %bb29
  ]

bb22:                                             ; preds = %bb20
  %37 = mul i32 %x, %w
  %38 = sdiv i32 %37, 4
  %.sum3 = add i32 %38, %.sum2
  %39 = add i32 %x, 15
  %40 = and i32 %39, -16
  %41 = add i32 %w, 15
  %42 = and i32 %41, -16
  %43 = mul i32 %40, %s
  %44 = icmp sgt i32 %x, 0
  br i1 %44, label %bb.nph, label %bb26

bb.nph:                                           ; preds = %bb22
  br label %bb23

bb23:                                             ; preds = %bb24, %bb.nph
  %y.21 = phi i32 [ 0, %bb.nph ], [ %indvar.next5, %bb24 ]
  %45 = mul i32 %y.21, %42
  %.sum1 = add i32 %45, %43
  %46 = getelementptr i8, i8* %r, i32 %.sum1
  %47 = mul i32 %y.21, %w
  %.sum5 = add i32 %47, %.sum3
  %48 = getelementptr i8, i8* %j, i32 %.sum5
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %48, i8* %46, i32 %w, i32 1, i1 false)
  br label %bb24

bb24:                                             ; preds = %bb23
  %indvar.next5 = add i32 %y.21, 1
  %exitcond6 = icmp ne i32 %indvar.next5, %x
  br i1 %exitcond6, label %bb23, label %bb24.bb26_crit_edge

bb24.bb26_crit_edge:                              ; preds = %bb24
  br label %bb26

bb26:                                             ; preds = %bb24.bb26_crit_edge, %bb22
  %49 = mul i32 %x, %w
  %.sum4 = add i32 %.sum3, %49
  %50 = getelementptr i8, i8* %j, i32 %.sum4
  %51 = mul i32 %x, %w
  %52 = sdiv i32 %51, 2
  tail call void @llvm.memset.p0i8.i32(i8* %50, i8 -128, i32 %52, i32 1, i1 false)
  ret void

bb29:                                             ; preds = %bb20, %entry
  %53 = add i32 %w, 15
  %54 = and i32 %53, -16
  %55 = icmp sgt i32 %x, 0
  br i1 %55, label %bb.nph11, label %bb33

bb.nph11:                                         ; preds = %bb29
  br label %bb30

bb30:                                             ; preds = %bb31, %bb.nph11
  %y.310 = phi i32 [ 0, %bb.nph11 ], [ %indvar.next13, %bb31 ]
  %56 = mul i32 %y.310, %54
  %57 = getelementptr i8, i8* %r, i32 %56
  %58 = mul i32 %y.310, %w
  %59 = getelementptr i8, i8* %j, i32 %58
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %59, i8* %57, i32 %w, i32 1, i1 false)
  br label %bb31

bb31:                                             ; preds = %bb30
  %indvar.next13 = add i32 %y.310, 1
  %exitcond14 = icmp ne i32 %indvar.next13, %x
  br i1 %exitcond14, label %bb30, label %bb31.bb33_crit_edge

bb31.bb33_crit_edge:                              ; preds = %bb31
  br label %bb33

bb33:                                             ; preds = %bb31.bb33_crit_edge, %bb29
  %60 = mul i32 %x, %w
  %61 = getelementptr i8, i8* %j, i32 %60
  %62 = mul i32 %x, %w
  %63 = sdiv i32 %62, 2
  tail call void @llvm.memset.p0i8.i32(i8* %61, i8 -128, i32 %63, i32 1, i1 false)
  ret void

return:                                           ; preds = %bb20
  ret void
}

define void @bar(i8* %r, i32 %s, i32 %w, i32 %x, i8* %j, i32 %d) nounwind {
entry:
  %0 = mul i32 %x, %w
  %1 = mul i32 %x, %w
  %2 = udiv i32 %1, 4
  %.sum2 = add i32 %2, %0
  %cond = icmp eq i32 %d, 1
  br i1 %cond, label %bb29, label %bb10.preheader

bb10.preheader:                                   ; preds = %entry
  %3 = icmp ne i32 %x, 0
  br i1 %3, label %bb.nph9, label %bb18.loopexit

bb.nph7:                                          ; preds = %bb7.preheader
  %4 = mul i32 %y.08, %w
  %5 = mul i32 %y.08, %s
  %6 = add i32 %5, 1
  %tmp8 = icmp ugt i32 1, %w
  %smax9 = select i1 %tmp8, i32 1, i32 %w
  br label %bb6

bb6:                                              ; preds = %bb7, %bb.nph7
  %x.06 = phi i32 [ 0, %bb.nph7 ], [ %indvar.next7, %bb7 ]
  %7 = add i32 %x.06, %4
  %8 = shl i32 %x.06, 1
  %9 = add i32 %6, %8
  %10 = getelementptr i8, i8* %r, i32 %9
  %11 = load i8, i8* %10, align 1
  %12 = getelementptr i8, i8* %j, i32 %7
  store i8 %11, i8* %12, align 1
  br label %bb7

bb7:                                              ; preds = %bb6
  %indvar.next7 = add i32 %x.06, 1
  %exitcond10 = icmp ne i32 %indvar.next7, %smax9
  br i1 %exitcond10, label %bb6, label %bb7.bb9_crit_edge

bb7.bb9_crit_edge:                                ; preds = %bb7
  br label %bb9

bb9:                                              ; preds = %bb7.preheader, %bb7.bb9_crit_edge
  br label %bb10

bb10:                                             ; preds = %bb9
  %indvar.next11 = add i32 %y.08, 1
  %exitcond12 = icmp ne i32 %indvar.next11, %x
  br i1 %exitcond12, label %bb7.preheader, label %bb10.bb18.loopexit_crit_edge

bb10.bb18.loopexit_crit_edge:                     ; preds = %bb10
  br label %bb10.bb18.loopexit_crit_edge.split

bb10.bb18.loopexit_crit_edge.split:               ; preds = %bb.nph9, %bb10.bb18.loopexit_crit_edge
  br label %bb18.loopexit

bb.nph9:                                          ; preds = %bb10.preheader
  %13 = icmp ugt i32 %w, 0
  br i1 %13, label %bb.nph9.split, label %bb10.bb18.loopexit_crit_edge.split

bb.nph9.split:                                    ; preds = %bb.nph9
  br label %bb7.preheader

bb7.preheader:                                    ; preds = %bb.nph9.split, %bb10
  %y.08 = phi i32 [ 0, %bb.nph9.split ], [ %indvar.next11, %bb10 ]
  br i1 true, label %bb.nph7, label %bb9

bb.nph5:                                          ; preds = %bb18.loopexit
  %14 = udiv i32 %w, 2
  %15 = icmp ult i32 %w, 2
  %16 = udiv i32 %x, 2
  br i1 %15, label %bb18.bb20_crit_edge.split, label %bb.nph5.split

bb.nph5.split:                                    ; preds = %bb.nph5
  %tmp2 = icmp ugt i32 1, %16
  %smax3 = select i1 %tmp2, i32 1, i32 %16
  br label %bb13

bb13:                                             ; preds = %bb18, %bb.nph5.split
  %y.14 = phi i32 [ 0, %bb.nph5.split ], [ %indvar.next1, %bb18 ]
  %17 = mul i32 %14, %y.14
  %18 = shl i32 %y.14, 1
  %19 = urem i32 %y.14, 2
  %20 = add i32 %19, %18
  %21 = mul i32 %20, %s
  br i1 true, label %bb.nph3, label %bb17

bb.nph3:                                          ; preds = %bb13
  %22 = add i32 %17, %0
  %23 = add i32 %17, %.sum2
  %24 = udiv i32 %w, 2
  %tmp = icmp ugt i32 1, %24
  %smax = select i1 %tmp, i32 1, i32 %24
  br label %bb14

bb14:                                             ; preds = %bb15, %bb.nph3
  %x.12 = phi i32 [ 0, %bb.nph3 ], [ %indvar.next, %bb15 ]
  %25 = shl i32 %x.12, 2
  %26 = add i32 %25, %21
  %27 = getelementptr i8, i8* %r, i32 %26
  %28 = load i8, i8* %27, align 1
  %.sum = add i32 %22, %x.12
  %29 = getelementptr i8, i8* %j, i32 %.sum
  store i8 %28, i8* %29, align 1
  %30 = shl i32 %x.12, 2
  %31 = or i32 %30, 2
  %32 = add i32 %31, %21
  %33 = getelementptr i8, i8* %r, i32 %32
  %34 = load i8, i8* %33, align 1
  %.sum6 = add i32 %23, %x.12
  %35 = getelementptr i8, i8* %j, i32 %.sum6
  store i8 %34, i8* %35, align 1
  br label %bb15

bb15:                                             ; preds = %bb14
  %indvar.next = add i32 %x.12, 1
  %exitcond = icmp ne i32 %indvar.next, %smax
  br i1 %exitcond, label %bb14, label %bb15.bb17_crit_edge

bb15.bb17_crit_edge:                              ; preds = %bb15
  br label %bb17

bb17:                                             ; preds = %bb15.bb17_crit_edge, %bb13
  br label %bb18

bb18.loopexit:                                    ; preds = %bb10.bb18.loopexit_crit_edge.split, %bb10.preheader
  %36 = icmp ult i32 %x, 2
  br i1 %36, label %bb20, label %bb.nph5

bb18:                                             ; preds = %bb17
  %indvar.next1 = add i32 %y.14, 1
  %exitcond4 = icmp ne i32 %indvar.next1, %smax3
  br i1 %exitcond4, label %bb13, label %bb18.bb20_crit_edge

bb18.bb20_crit_edge:                              ; preds = %bb18
  br label %bb18.bb20_crit_edge.split

bb18.bb20_crit_edge.split:                        ; preds = %bb18.bb20_crit_edge, %bb.nph5
  br label %bb20

bb20:                                             ; preds = %bb18.bb20_crit_edge.split, %bb18.loopexit
  switch i32 %d, label %return [
    i32 3, label %bb22
    i32 1, label %bb29
  ]

bb22:                                             ; preds = %bb20
  %37 = mul i32 %x, %w
  %38 = udiv i32 %37, 4
  %.sum3 = add i32 %38, %.sum2
  %39 = add i32 %x, 15
  %40 = and i32 %39, -16
  %41 = add i32 %w, 15
  %42 = and i32 %41, -16
  %43 = mul i32 %40, %s
  %44 = icmp ugt i32 %x, 0
  br i1 %44, label %bb.nph, label %bb26

bb.nph:                                           ; preds = %bb22
  br label %bb23

bb23:                                             ; preds = %bb24, %bb.nph
  %y.21 = phi i32 [ 0, %bb.nph ], [ %indvar.next5, %bb24 ]
  %45 = mul i32 %y.21, %42
  %.sum1 = add i32 %45, %43
  %46 = getelementptr i8, i8* %r, i32 %.sum1
  %47 = mul i32 %y.21, %w
  %.sum5 = add i32 %47, %.sum3
  %48 = getelementptr i8, i8* %j, i32 %.sum5
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %48, i8* %46, i32 %w, i32 1, i1 false)
  br label %bb24

bb24:                                             ; preds = %bb23
  %indvar.next5 = add i32 %y.21, 1
  %exitcond6 = icmp ne i32 %indvar.next5, %x
  br i1 %exitcond6, label %bb23, label %bb24.bb26_crit_edge

bb24.bb26_crit_edge:                              ; preds = %bb24
  br label %bb26

bb26:                                             ; preds = %bb24.bb26_crit_edge, %bb22
  %49 = mul i32 %x, %w
  %.sum4 = add i32 %.sum3, %49
  %50 = getelementptr i8, i8* %j, i32 %.sum4
  %51 = mul i32 %x, %w
  %52 = udiv i32 %51, 2
  tail call void @llvm.memset.p0i8.i32(i8* %50, i8 -128, i32 %52, i32 1, i1 false)
  ret void

bb29:                                             ; preds = %bb20, %entry
  %53 = add i32 %w, 15
  %54 = and i32 %53, -16
  %55 = icmp ugt i32 %x, 0
  br i1 %55, label %bb.nph11, label %bb33

bb.nph11:                                         ; preds = %bb29
  br label %bb30

bb30:                                             ; preds = %bb31, %bb.nph11
  %y.310 = phi i32 [ 0, %bb.nph11 ], [ %indvar.next13, %bb31 ]
  %56 = mul i32 %y.310, %54
  %57 = getelementptr i8, i8* %r, i32 %56
  %58 = mul i32 %y.310, %w
  %59 = getelementptr i8, i8* %j, i32 %58
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %59, i8* %57, i32 %w, i32 1, i1 false)
  br label %bb31

bb31:                                             ; preds = %bb30
  %indvar.next13 = add i32 %y.310, 1
  %exitcond14 = icmp ne i32 %indvar.next13, %x
  br i1 %exitcond14, label %bb30, label %bb31.bb33_crit_edge

bb31.bb33_crit_edge:                              ; preds = %bb31
  br label %bb33

bb33:                                             ; preds = %bb31.bb33_crit_edge, %bb29
  %60 = mul i32 %x, %w
  %61 = getelementptr i8, i8* %j, i32 %60
  %62 = mul i32 %x, %w
  %63 = udiv i32 %62, 2
  tail call void @llvm.memset.p0i8.i32(i8* %61, i8 -128, i32 %63, i32 1, i1 false)
  ret void

return:                                           ; preds = %bb20
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind
