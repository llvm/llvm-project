; RUN: opt -consthoist -consthoist-gep -S -o - %s | FileCheck %s

target triple = "aarch64-none--musleabi"

; Check that constant GEP expressions are rewritten to one-dimensional
; (single-index) GEPs, whose base poiner is a multi-dimensional GEP.
; CHECK:  %const = bitcast i32* getelementptr inbounds (%0, %0* @global, i32 0, i32 4, i32 0, i32 0) to i32*
; CHECK-NEXT:  store i32 undef, i32* %const, align 4

; CHECK-NEXT:  %[[BC1:[a-z0-9_]+]] = bitcast i32* %const to i8*
; CHECK-NEXT:  %[[M1:[a-z0-9_]+]] = getelementptr i8, i8* %[[BC1]], i32 4
; CHECK-NEXT:  %[[BC2:[a-z0-9_]+]] = bitcast i8* %[[M1]] to i32*
; CHECK-NEXT:  store i32 undef, i32* %[[BC2]], align 4

; CHECK-NEXT:  %[[BC3:[a-z0-9_]+]] = bitcast i32* %const to i8*
; CHECK-NEXT:  %[[M2:[a-z0-9_]+]] = getelementptr i8, i8* %[[BC3]], i32 160
; CHECK-NEXT:  %[[BC4:[a-z0-9_]+]] = bitcast i8* %[[M2]] to i32*
; CHECK-NEXT:  store i32 undef, i32* %[[BC4]], align 4

; CHECK-NEXT:  %[[BC5:[a-z0-9_]+]] = bitcast i32* %const to i8*
; CHECK-NEXT:  %[[M3:[a-z0-9_]+]] = getelementptr i8, i8* %[[BC5]], i32 164
; CHECK-NEXT:  %[[BC6:[a-z0-9_]+]] = bitcast i8* %[[M3]] to i32*
; CHECK-NEXT:  store i32 undef, i32* %[[BC6]], align 4

%0 = type { %1, %2, [9 x i16], %6, %7 }
%1 = type { i32, i32, i32, i32, i32, i32, i16, i16, i8, i8, i16, i32, i32, i16, i8, i8 }
%2 = type { i32, %3, i8, i8, i8, i8, i32, %4, %5, [16 x i8], i16, i16, i8, i8, i8, i8, i32, i32, i32 }
%3 = type { i16, i8, i8 }
%4 = type { i16, i8, i8 }
%5 = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
%6 = type { i8, i8 }
%7 = type { [5 x i32], [3 x i32], [6 x i32], [3 x i32], [2 x i32], [4 x i32], [3 x i32], [2 x i32], [4 x i32], [5 x i32], [3 x i32], [6 x i32], [1 x i32], i32, i32, i32, i32, i32, i32 }

@global = external dso_local local_unnamed_addr global %0, align 4

define dso_local void @zot() {
bb:
  store i32 undef, i32* getelementptr inbounds (%0, %0* @global, i32 0, i32 4, i32 0, i32 0), align 4
  store i32 undef, i32* getelementptr inbounds (%0, %0* @global, i32 0, i32 4, i32 0, i32 1), align 4
  store i32 undef, i32* getelementptr inbounds (%0, %0* @global, i32 0, i32 4, i32 11, i32 0), align 4
  store i32 undef, i32* getelementptr inbounds (%0, %0* @global, i32 0, i32 4, i32 11, i32 1), align 4
  ret void
}

