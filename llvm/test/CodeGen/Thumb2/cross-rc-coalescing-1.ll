; RUN: llc < %s -mtriple=thumbv7-apple-darwin9 -mcpu=cortex-a8

%struct.FILE = type { ptr, i32, i32, i16, i16, %struct.__sbuf, i32, ptr, ptr, ptr, ptr, ptr, %struct.__sbuf, ptr, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct.__sFILEX = type opaque
%struct.__sbuf = type { ptr, i32 }

declare i32 @fgetc(ptr nocapture) nounwind

define i32 @main(i32 %argc, ptr nocapture %argv) nounwind {
entry:
  br i1 undef, label %bb, label %bb1

bb:                                               ; preds = %entry
  unreachable

bb1:                                              ; preds = %entry
  br i1 undef, label %bb.i1, label %bb1.i2

bb.i1:                                            ; preds = %bb1
  unreachable

bb1.i2:                                           ; preds = %bb1
  %0 = call  i32 @fgetc(ptr undef) nounwind ; <i32> [#uses=0]
  br i1 undef, label %bb2.i3, label %bb3.i4

bb2.i3:                                           ; preds = %bb1.i2
  br i1 undef, label %bb4.i, label %bb3.i4

bb3.i4:                                           ; preds = %bb2.i3, %bb1.i2
  unreachable

bb4.i:                                            ; preds = %bb2.i3
  br i1 undef, label %bb5.i, label %get_image.exit

bb5.i:                                            ; preds = %bb4.i
  unreachable

get_image.exit:                                   ; preds = %bb4.i
  br i1 undef, label %bb28, label %bb27

bb27:                                             ; preds = %get_image.exit
  br label %bb.i

bb.i:                                             ; preds = %bb.i, %bb27
  %1 = fptrunc double undef to float              ; <float> [#uses=1]
  %2 = fptoui float %1 to i8                      ; <i8> [#uses=1]
  store i8 %2, ptr undef, align 1
  br label %bb.i

bb28:                                             ; preds = %get_image.exit
  unreachable
}
