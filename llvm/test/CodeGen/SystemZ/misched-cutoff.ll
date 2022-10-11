; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 -misched-cutoff=1 -o /dev/null < %s
; REQUIRES: asserts
; -misched=shuffle isn't available in NDEBUG builds!

; Test that the post-ra scheduler does not crash with -misched-cutoff.

@g_184 = external dso_local global i16, align 2
@g_294 = external dso_local global [1 x [9 x ptr]], align 8

define void @fun() {
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %i = phi i64 [ 0, %bb ], [ %i22, %bb1 ]
  %i2 = trunc i64 %i to i32
  %i3 = lshr i32 %i2, 1
  %i4 = select i1 false, i32 %i3, i32 undef
  %i5 = lshr i32 %i4, 1
  %i6 = xor i32 %i5, -306674912
  %i7 = select i1 undef, i32 %i5, i32 %i6
  %i8 = lshr i32 %i7, 1
  %i9 = xor i32 %i8, -306674912
  %i10 = select i1 undef, i32 %i8, i32 %i9
  %i11 = lshr i32 %i10, 1
  %i12 = xor i32 %i11, -306674912
  %i13 = select i1 undef, i32 %i11, i32 %i12
  %i14 = lshr i32 %i13, 1
  %i15 = select i1 false, i32 %i14, i32 undef
  %i16 = lshr i32 %i15, 1
  %i17 = select i1 false, i32 %i16, i32 undef
  %i18 = lshr i32 %i17, 1
  %i19 = select i1 false, i32 %i18, i32 undef
  %i20 = lshr i32 %i19, 1
  %i21 = select i1 false, i32 %i20, i32 undef
  store i32 %i21, ptr undef, align 4
  %i22 = add nuw nsw i64 %i, 1
  %i23 = icmp ult i64 %i, 255
  br i1 %i23, label %bb1, label %bb24

bb24:                                             ; preds = %bb1
  %i25 = load volatile i16, ptr undef
  store ptr null, ptr undef, align 8
  store i32 -10, ptr undef, align 4
  store i32 -10, ptr null, align 4
  store i32 -10, ptr undef, align 4
  store i16 0, ptr @g_184, align 2
  store ptr null, ptr getelementptr inbounds ([1 x [9 x ptr]], ptr @g_294, i64 0, i64 0, i64 2), align 8
  store ptr null, ptr getelementptr inbounds ([1 x [9 x ptr]], ptr @g_294, i64 0, i64 0, i64 5), align 8
  unreachable
}
