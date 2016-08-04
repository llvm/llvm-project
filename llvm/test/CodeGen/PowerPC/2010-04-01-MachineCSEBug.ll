; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin10.0
; rdar://7819990

%0 = type { i32 }
%1 = type { i64 }
%struct.Buffer = type { [1024 x i8], i64, i64, i64 }
%struct.InStream = type { %struct.Buffer, %0, %1, i32*, %struct.InStreamMethods* }
%struct.InStreamMethods = type { void (%struct.InStream*, i8*, i32)*, void (%struct.InStream*, i64)*, i64 (%struct.InStream*)*, void (%struct.InStream*)* }

define i64 @t(%struct.InStream* %is) nounwind optsize ssp {
entry:
  br i1 undef, label %is_read_byte.exit, label %bb.i

bb.i:                                             ; preds = %entry
  br label %is_read_byte.exit

is_read_byte.exit:                                ; preds = %bb.i, %entry
  br i1 undef, label %is_read_byte.exit22, label %bb.i21

bb.i21:                                           ; preds = %is_read_byte.exit
  unreachable

is_read_byte.exit22:                              ; preds = %is_read_byte.exit
  br i1 undef, label %is_read_byte.exit19, label %bb.i18

bb.i18:                                           ; preds = %is_read_byte.exit22
  br label %is_read_byte.exit19

is_read_byte.exit19:                              ; preds = %bb.i18, %is_read_byte.exit22
  br i1 undef, label %is_read_byte.exit16, label %bb.i15

bb.i15:                                           ; preds = %is_read_byte.exit19
  unreachable

is_read_byte.exit16:                              ; preds = %is_read_byte.exit19
  %0 = shl i64 undef, 32                          ; <i64> [#uses=1]
  br i1 undef, label %is_read_byte.exit13, label %bb.i12

bb.i12:                                           ; preds = %is_read_byte.exit16
  unreachable

is_read_byte.exit13:                              ; preds = %is_read_byte.exit16
  %1 = shl i64 undef, 24                          ; <i64> [#uses=1]
  br i1 undef, label %is_read_byte.exit10, label %bb.i9

bb.i9:                                            ; preds = %is_read_byte.exit13
  unreachable

is_read_byte.exit10:                              ; preds = %is_read_byte.exit13
  %2 = shl i64 undef, 16                          ; <i64> [#uses=1]
  br i1 undef, label %is_read_byte.exit7, label %bb.i6

bb.i6:                                            ; preds = %is_read_byte.exit10
  br label %is_read_byte.exit7

is_read_byte.exit7:                               ; preds = %bb.i6, %is_read_byte.exit10
  %3 = shl i64 undef, 8                           ; <i64> [#uses=1]
  br i1 undef, label %is_read_byte.exit4, label %bb.i3

bb.i3:                                            ; preds = %is_read_byte.exit7
  unreachable

is_read_byte.exit4:                               ; preds = %is_read_byte.exit7
  %4 = or i64 0, %0                               ; <i64> [#uses=1]
  %5 = or i64 %4, %1                              ; <i64> [#uses=1]
  %6 = or i64 %5, %2                              ; <i64> [#uses=1]
  %7 = or i64 %6, %3                              ; <i64> [#uses=1]
  %8 = or i64 %7, 0                               ; <i64> [#uses=1]
  ret i64 %8
}
