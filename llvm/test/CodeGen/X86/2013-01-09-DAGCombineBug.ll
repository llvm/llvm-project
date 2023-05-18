; RUN: llc -mtriple=x86_64-apple-macosx10.5.0 < %s

; rdar://12968664

define void @t() nounwind uwtable ssp {
  br label %4

; <label>:1                                       ; preds = %4, %2
  ret void

; <label>:2                                       ; preds = %6, %5, %3, %2
  switch i32 undef, label %2 [
    i32 1090573978, label %1
    i32 1090573938, label %3
    i32 1090573957, label %5
  ]

; <label>:3                                       ; preds = %4, %2
  br i1 undef, label %2, label %4

; <label>:4                                       ; preds = %6, %5, %3, %0
  switch i32 undef, label %11 [
    i32 1090573938, label %3
    i32 1090573957, label %5
    i32 1090573978, label %1
    i32 165205179, label %6
  ]

; <label>:5                                       ; preds = %4, %2
  br i1 undef, label %2, label %4

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 undef, 590901838
  %8 = or i1 false, %7
  %9 = or i1 true, %8
  %10 = xor i1 %8, %9
  br i1 %10, label %4, label %2

; <label>:11                                      ; preds = %11, %4
  br label %11
}

; PR15608
@global = external constant [2 x i8]

define void @PR15608() {
bb:
  br label %bb3

bb1:                                              ; No predecessors!
  %constexpr = ptrtoint ptr @global to i64
  %constexpr1 = zext i64 %constexpr to i384
  %constexpr2 = shl i384 %constexpr1, 192
  %constexpr3 = or i384 %constexpr2, 425269881901436522087161771558896140289
  %constexpr4 = lshr i384 %constexpr3, 128
  %constexpr5 = trunc i384 %constexpr4 to i128
  %constexpr6 = lshr i128 %constexpr5, 64
  %constexpr7 = trunc i128 %constexpr6 to i64
  %constexpr8 = zext i64 %constexpr7 to i192
  %constexpr9 = shl i192 %constexpr8, 64
  %constexpr10 = or i192 %constexpr9, 1
  %constexpr11 = lshr i192 %constexpr10, 128
  %constexpr12 = trunc i192 %constexpr11 to i1
  %constexpr13 = zext i1 %constexpr12 to i64
  %constexpr14 = xor i64 %constexpr13, 1
  %constexpr15 = icmp ult i64 %constexpr14, 1
  br i1 %constexpr15, label %bb2, label %bb3

bb2:                                              ; preds = %bb1
  unreachable

bb3:                                              ; preds = %bb1, %bb
  %constexpr16 = ptrtoint ptr @global to i64
  %constexpr17 = zext i64 %constexpr16 to i384
  %constexpr18 = shl i384 %constexpr17, 192
  %constexpr19 = or i384 %constexpr18, 425269881901436522087161771558896140289
  %constexpr20 = lshr i384 %constexpr19, 128
  %constexpr21 = trunc i384 %constexpr20 to i128
  %constexpr22 = lshr i128 %constexpr21, 64
  %constexpr23 = trunc i128 %constexpr22 to i64
  %constexpr24 = zext i64 %constexpr23 to i192
  %constexpr25 = shl i192 %constexpr24, 64
  %constexpr26 = or i192 %constexpr25, 1
  %constexpr27 = lshr i192 %constexpr26, 128
  %constexpr28 = trunc i192 %constexpr27 to i1
  %constexpr29 = and i192 %constexpr26, -340282366920938463463374607431768211457
  %constexpr30 = zext i1 %constexpr28 to i192
  %constexpr31 = shl i192 %constexpr30, 128
  %constexpr32 = or i192 %constexpr29, %constexpr31
  %constexpr33 = lshr i192 %constexpr32, 128
  %constexpr34 = trunc i192 %constexpr33 to i1
  %constexpr35 = xor i1 %constexpr28, %constexpr34
  br i1 %constexpr35, label %bb7, label %phi.constexpr

phi.constexpr:                                    ; preds = %bb3
  %constexpr36 = ptrtoint ptr @global to i64
  %constexpr37 = zext i64 %constexpr36 to i384
  %constexpr38 = shl i384 %constexpr37, 192
  %constexpr39 = or i384 %constexpr38, 425269881901436522087161771558896140289
  %constexpr40 = lshr i384 %constexpr39, 128
  %constexpr41 = trunc i384 %constexpr40 to i128
  %constexpr42 = lshr i128 %constexpr41, 64
  %constexpr43 = trunc i128 %constexpr42 to i64
  %constexpr44 = zext i64 %constexpr43 to i192
  %constexpr45 = shl i192 %constexpr44, 64
  %constexpr46 = or i192 %constexpr45, 1
  %constexpr47 = and i192 %constexpr46, -340282366920938463463374607431768211457
  %constexpr48 = lshr i192 %constexpr46, 128
  %constexpr49 = trunc i192 %constexpr48 to i1
  %constexpr50 = zext i1 %constexpr49 to i192
  %constexpr51 = shl i192 %constexpr50, 128
  %constexpr52 = or i192 %constexpr47, %constexpr51
  %constexpr53 = lshr i192 %constexpr52, 128
  %constexpr54 = trunc i192 %constexpr53 to i1
  br label %bb4

bb4:                                              ; preds = %phi.constexpr, %bb6
  %tmp = phi i1 [ true, %bb6 ], [ %constexpr54, %phi.constexpr ]
  br i1 false, label %bb8, label %bb5

bb5:                                              ; preds = %bb4
  br i1 %tmp, label %bb8, label %bb6

bb6:                                              ; preds = %bb5
  br i1 false, label %bb8, label %bb4

bb7:                                              ; preds = %bb3
  unreachable

bb8:                                              ; preds = %bb6, %bb5, %bb4
  unreachable
}
