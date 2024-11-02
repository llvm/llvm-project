; RUN: opt -passes=consthoist -consthoist-gep -S -o - %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-none--musleabi"

; Check that for the same offset from the base constant, different types are materialized separately.
; CHECK: %const = bitcast %5** getelementptr inbounds (%0, %0* @global, i32 0, i32 2, i32 0) to %5**
; CHECK: %tmp = load %5*, %5** %const, align 4
; CHECK: %base_bitcast = bitcast %5** %const to i8*
; CHECK: %mat_gep = getelementptr i8, i8* %base_bitcast, i32 0
; CHECK: %mat_bitcast = bitcast i8* %mat_gep to %4*
; CHECK: tail call void undef(%5* nonnull %tmp, %4* %mat_bitcast)

%0 = type { [16 x %1], %2, %4, [16 x %5], %6, %7, i32, [4 x i32], [8 x %3], i8, i8, i8, i8, i8, i8, i8, %8, %11, %11*, i32, i16, i8, i8, i8, i8, i8, i8, [15 x i16], i8, i8, [23 x %12], i8, i8*, i8, %13, i8, i8 }
%1 = type { i32, i32, i8, i8, i8, i8, i8, i8, i8, i8 }
%2 = type { %3*, i16, i16, i16 }
%3 = type { [4 x i32] }
%4 = type { %5*, %5*, i8 }
%5 = type { [4 x i32], i8*, i8, i8 }
%6 = type { i8, [4 x i32] }
%7 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%8 = type { [16 x %9], %9*, %9*, %9*, %9*, %11, %11, %11, i8, i8, i8, i8 }
%9 = type { %1, %11, %11, %9*, %9*, %10, i8, i8, i8, i8 }
%10 = type { i32, i16 }
%11 = type { %11*, %11* }
%12 = type { i8, i16, i32 }
%13 = type { i32, i32, i8 }

@global = external dso_local global %0, align 4

; Function Attrs: nounwind optsize ssp
define dso_local void @zot() {
bb:
  br i1 undef, label %bb2, label %bb1

bb1:                                              ; preds = %bb
  %tmp = load %5*, %5** getelementptr inbounds (%0, %0* @global, i32 0, i32 2, i32 0), align 4
  tail call void undef(%5* nonnull %tmp, %4* getelementptr inbounds (%0, %0* @global, i32 0, i32 2))
  unreachable

bb2:                                              ; preds = %bb
  ret void
}

