; RUN: opt -passes=consthoist -consthoist-gep -S -o - %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-none--musleabi"

; Check that for the same offset from the base constant, different types are materialized separately.
; CHECK: %const = bitcast ptr getelementptr inbounds (%0, ptr @global, i32 0, i32 2, i32 0) to ptr
; CHECK: %tmp = load ptr, ptr %const, align 4
; CHECK: tail call void undef(ptr nonnull %tmp, ptr %const)

%0 = type { [16 x %1], %2, %4, [16 x %5], %6, %7, i32, [4 x i32], [8 x %3], i8, i8, i8, i8, i8, i8, i8, %8, %11, ptr, i32, i16, i8, i8, i8, i8, i8, i8, [15 x i16], i8, i8, [23 x %12], i8, ptr, i8, %13, i8, i8 }
%1 = type { i32, i32, i8, i8, i8, i8, i8, i8, i8, i8 }
%2 = type { ptr, i16, i16, i16 }
%3 = type { [4 x i32] }
%4 = type { ptr, ptr, i8 }
%5 = type { [4 x i32], ptr, i8, i8 }
%6 = type { i8, [4 x i32] }
%7 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%8 = type { [16 x %9], ptr, ptr, ptr, ptr, %11, %11, %11, i8, i8, i8, i8 }
%9 = type { %1, %11, %11, ptr, ptr, %10, i8, i8, i8, i8 }
%10 = type { i32, i16 }
%11 = type { ptr, ptr }
%12 = type { i8, i16, i32 }
%13 = type { i32, i32, i8 }

@global = external dso_local global %0, align 4

; Function Attrs: nounwind optsize ssp
define dso_local void @zot() {
bb:
  br i1 undef, label %bb2, label %bb1

bb1:                                              ; preds = %bb
  %tmp = load ptr, ptr getelementptr inbounds (%0, ptr @global, i32 0, i32 2, i32 0), align 4
  tail call void undef(ptr nonnull %tmp, ptr getelementptr inbounds (%0, ptr @global, i32 0, i32 2))
  unreachable

bb2:                                              ; preds = %bb
  ret void
}

