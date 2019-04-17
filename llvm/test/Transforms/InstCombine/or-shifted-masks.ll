; RUN: opt -S -instcombine < %s | FileCheck %s

define i32 @or_and_shifts1(i32 %x) {
; CHECK-LABEL: @or_and_shifts1(
; CHECK-NEXT:    [[TMP1:%.*]] = shl i32 %x, 3
; CHECK-NEXT:    [[TMP2:%.*]] = and i32 [[TMP1]], 8
; CHECK-NEXT:    [[TMP3:%.*]] = shl i32 %x, 5
; CHECK-NEXT:    [[TMP4:%.*]] = and i32 [[TMP3]], 32
; CHECK-NEXT:    [[TMP5:%.*]] = or i32 [[TMP2]], [[TMP4]]
; CHECK-NEXT:    ret i32 [[TMP5]]
;
  %1 = shl i32 %x, 3
  %2 = and i32 %1, 15
  %3 = shl i32 %x, 5
  %4 = and i32 %3, 60
  %5 = or i32 %2, %4
  ret i32 %5
}

define i32 @or_and_shifts2(i32 %x) {
; CHECK-LABEL: @or_and_shifts2(
; CHECK-NEXT:    [[TMP1:%.*]] = shl i32 %x, 3
; CHECK-NEXT:    [[TMP2:%.*]] = and i32 [[TMP1]], 896
; CHECK-NEXT:    [[TMP3:%.*]] = lshr i32 %x, 4
; CHECK-NEXT:    [[TMP4:%.*]] = and i32 [[TMP3]], 7
; CHECK-NEXT:    [[TMP5:%.*]] = or i32 [[TMP2]], [[TMP4]]
; CHECK-NEXT:    ret i32 [[TMP5]]
;
  %1 = shl i32 %x, 3
  %2 = and i32 %1, 896
  %3 = lshr i32 %x, 4
  %4 = and i32 %3, 7
  %5 = or i32 %2, %4
  ret i32 %5
}

define i32 @or_and_shift_shift_and(i32 %x) {
; CHECK-LABEL: @or_and_shift_shift_and(
; CHECK-NEXT:    [[TMP1:%.*]] = shl i32 %x, 3
; CHECK-NEXT:    [[TMP2:%.*]] = and i32 [[TMP1]], 56
; CHECK-NEXT:    [[TMP3:%.*]] = shl i32 %x, 2
; CHECK-NEXT:    [[TMP4:%.*]] = and i32 [[TMP3]], 28
; CHECK-NEXT:    [[TMP5:%.*]] = or i32 [[TMP2]], [[TMP4]]
; CHECK-NEXT:    ret i32 [[TMP5]]
;
  %1 = and i32 %x, 7
  %2 = shl i32 %1, 3
  %3 = shl i32 %x, 2
  %4 = and i32 %3, 28
  %5 = or i32 %2, %4
  ret i32 %5
}

define i32 @multiuse1(i32 %x) {
; CHECK-LABEL: @multiuse1(
; CHECK-NEXT:    [[TMP1:%.*]] = shl i32 %x, 6
; CHECK-NEXT:    [[TMP2:%.*]] = and i32 [[TMP1]], 384
; CHECK-NEXT:    [[TMP3:%.*]] = lshr i32 %x, 1
; CHECK-NEXT:    [[TMP4:%.*]] = and i32 [[TMP3]], 3
; CHECK-NEXT:    [[TMP5:%.*]] = or i32 [[TMP4]], [[TMP2]]
; CHECK-NEXT:    ret i32 [[TMP5]]
;
  %1 = and i32 %x, 2
  %2 = and i32 %x, 4
  %3 = shl nuw nsw i32 %1, 6
  %4 = lshr exact i32 %1, 1
  %5 = shl nuw nsw i32 %2, 6
  %6 = lshr exact i32 %2, 1
  %7 = or i32 %3, %5
  %8 = or i32 %4, %6
  %9 = or i32 %8, %7
  ret i32 %9
}

define i32 @multiuse2(i32 %x) {
; CHECK-LABEL: @multiuse2(
; CHECK-NEXT:    [[TMP1:%.*]] = shl i32 %x, 1
; CHECK-NEXT:    [[TMP2:%.*]] = and i32 [[TMP1]], 12
; CHECK-NEXT:    [[TMP3:%.*]] = shl i32 %x, 8
; CHECK-NEXT:    [[TMP4:%.*]] = and i32 [[TMP3]], 24576
; CHECK-NEXT:    [[TMP5:%.*]] = shl i32 %x, 8
; CHECK-NEXT:    [[TMP6:%.*]] = and i32 [[TMP5]], 7680
; CHECK-NEXT:    [[TMP7:%.*]] = or i32 [[TMP4]], [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = shl i32 %x, 1
; CHECK-NEXT:    [[TMP9:%.*]] = and i32 [[TMP8]], 240
; CHECK-NEXT:    [[TMP10:%.*]] = or i32 [[TMP2]], [[TMP9]]
; CHECK-NEXT:    [[TMP11:%.*]] = or i32 [[TMP7]], [[TMP10]]
; CHECK-NEXT:    ret i32 [[TMP11]]
;
  %1 = and i32 %x, 6
  %2 = shl nuw nsw i32 %1, 8
  %3 = shl nuw nsw i32 %1, 1
  %4 = and i32 %x, 24
  %5 = shl nuw nsw i32 %4, 8
  %6 = shl nuw nsw i32 %4, 1
  %7 = and i32 %x, 96
  %8 = shl nuw nsw i32 %7, 8
  %9 = shl nuw nsw i32 %7, 1
  %10 = or i32 %2, %5
  %11 = or i32 %8, %10
  %12 = or i32 %9, %6
  %13 = or i32 %3, %12
  %14 = or i32 %11, %13
  ret i32 %14
}

define i32 @multiuse3(i32 %x) {
; CHECK-LABEL: @multiuse3(
; CHECK-NEXT:    [[TMP1:%.*]] = and i32 %x, 96
; CHECK-NEXT:    [[TMP2:%.*]] = shl nuw nsw i32 [[TMP1]], 6
; CHECK-NEXT:    [[TMP3:%.*]] = lshr exact i32 [[TMP1]], 1
; CHECK-NEXT:    [[TMP4:%.*]] = shl i32 %x, 6
; CHECK-NEXT:    [[TMP5:%.*]] = and i32 [[TMP4]], 1920
; CHECK-NEXT:    [[TMP6:%.*]] = or i32 [[TMP2]], [[TMP5]]
; CHECK-NEXT:    [[TMP7:%.*]] = lshr i32 %x, 1
; CHECK-NEXT:    [[TMP8:%.*]] = and i32 [[TMP7]], 15
; CHECK-NEXT:    [[TMP9:%.*]] = or i32 [[TMP3]], [[TMP8]]
; CHECK-NEXT:    [[TMP10:%.*]] = or i32 [[TMP9]], [[TMP6]]
; CHECK-NEXT:    ret i32 [[TMP10]]
;
  %1 = and i32 %x, 96
  %2 = shl nuw nsw i32 %1, 6
  %3 = lshr exact i32 %1, 1
  %4 = shl i32 %x, 6
  %5 = and i32 %4, 1920
  %6 = or i32 %2, %5
  %7 = lshr i32 %x, 1
  %8 = and i32 %7, 15
  %9 = or i32 %3, %8
  %10 = or i32 %9, %6
  ret i32 %10
}

define i32 @multiuse4(i32 %x) local_unnamed_addr #0 {
; CHECK-LABEL: @multiuse4(
; CHECK-NEXT:    [[TMP1:%.*]] = and i32 %x, 100663296
; CHECK-NEXT:    [[TMP2:%.*]] = icmp sgt i32 %x, -1
; CHECK-NEXT:    br i1 [[TMP2]], label %if, label %else
; CHECK:         {{.*}}if:{{.*}}
; CHECK-NEXT:    [[TMP3:%.*]] = lshr exact i32 [[TMP1]], 22
; CHECK-NEXT:    [[TMP4:%.*]] = lshr i32 %x, 22
; CHECK-NEXT:    [[TMP5:%.*]] = and i32 [[TMP4]], 480
; CHECK-NEXT:    [[TMP6:%.*]] = or i32 [[TMP5]], [[TMP3]]
; CHECK-NEXT:    br label %end
; CHECK:         {{.*}}else:{{.*}}
; CHECK-NEXT:    [[TMP7:%.*]] = lshr exact i32 [[TMP1]], 17
; CHECK-NEXT:    [[TMP8:%.*]] = lshr i32 %x, 17
; CHECK-NEXT:    [[TMP9:%.*]] = and i32 [[TMP8]], 15360
; CHECK-NEXT:    [[TMP10:%.*]] = or i32 [[TMP9]], [[TMP7]]
; CHECK-NEXT:    br label %end
; CHECK:         {{.*}}end{{.*}}
; CHECK-NEXT:    [[TMP11:%.*]] = phi i32 [ [[TMP6]], %if ], [ [[TMP10]], %else ]
; CHECK-NEXT:    ret i32 [[TMP11]]
;
  %1 = and i32 %x, 100663296
  %2 = icmp sgt i32 %x, -1
  br i1 %2, label %if, label %else

if:
  %3 = lshr exact i32 %1, 22
  %4 = lshr i32 %x, 22
  %5 = and i32 %4, 480
  %6 = or i32 %5, %3
  br label %end

else:
  %7 = lshr exact i32 %1, 17
  %8 = lshr i32 %x, 17
  %9 = and i32 %8, 15360
  %10 = or i32 %9, %7
  br label %end

end:
  %11 = phi i32 [ %6, %if ], [ %10, %else ]
  ret i32 %11
}

define i32 @multiuse5(i32 %x) local_unnamed_addr #0 {
; CHECK-LABEL: @multiuse5(
; CHECK-NEXT:    [[TMP1:%.*]] = shl i32 %x, 5
; CHECK-NEXT:    [[TMP2:%.*]] = icmp sgt i32 %x, -1
; CHECK-NEXT:    br i1 [[TMP2]], label %if, label %else
; CHECK:         {{.*}}if:{{.*}}
; CHECK-NEXT:    [[TMP3:%.*]] = and i32 [[TMP1]], 21760
; CHECK-NEXT:    [[TMP4:%.*]] = shl i32 %x, 5
; CHECK-NEXT:    [[TMP5:%.*]] = and i32 [[TMP4]], 43520
; CHECK-NEXT:    [[TMP6:%.*]] = or i32 [[TMP5]], [[TMP3]]
; CHECK-NEXT:    br label %end
; CHECK:         {{.*}}else:{{.*}}
; CHECK-NEXT:    [[TMP7:%.*]] = and i32 [[TMP1]], 5570560
; CHECK-NEXT:    [[TMP8:%.*]] = shl i32 %x, 5
; CHECK-NEXT:    [[TMP9:%.*]] = and i32 [[TMP8]], 11141120
; CHECK-NEXT:    [[TMP10:%.*]] = or i32 [[TMP9]], [[TMP7]]
; CHECK-NEXT:    br label %end
; CHECK:         {{.*}}end{{.*}}
; CHECK-NEXT:    [[TMP11:%.*]] = phi i32 [ [[TMP6]], %if ], [ [[TMP10]], %else ]
; CHECK-NEXT:    ret i32 [[TMP11]]
;
  %1 = shl i32 %x, 5
  %2 = icmp sgt i32 %x, -1
  br i1 %2, label %if, label %else

if:
  %3 = and i32 %1, 21760
  %4 = and i32 %x, 1360
  %5 = shl nuw nsw i32 %4, 5
  %6 = or i32 %5, %3
  br label %end

else:
  %7 = and i32 %1, 5570560
  %8 = and i32 %x, 348160
  %9 = shl nuw nsw i32 %8, 5
  %10 = or i32 %9, %7
  br label %end

end:
  %11 = phi i32 [ %6, %if ], [ %10, %else ]
  ret i32 %11
}

