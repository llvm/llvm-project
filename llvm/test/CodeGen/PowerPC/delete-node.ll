; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--

; The DAGCombiner leaves behind a dead node in this testcase. Currently
; ISel is ignoring dead nodes, though it would be preferable for
; DAGCombiner to be able to eliminate the dead node.

define void @GrayATo32ARGBTabB(ptr %baseAddr, ptr %cmp, i32 %rowBytes) nounwind {
entry:
      	br label %bb1

bb1:            ; preds = %bb1, %entry
        %0 = load i16, ptr null, align 2            ; <i16> [#uses=1]
        %1 = ashr i16 %0, 4             ; <i16> [#uses=1]
        %2 = sext i16 %1 to i32         ; <i32> [#uses=1]
        %3 = getelementptr i8, ptr null, i32 %2             ; <ptr> [#uses=1]
        %4 = load i8, ptr %3, align 1               ; <i8> [#uses=1]
        %5 = zext i8 %4 to i32          ; <i32> [#uses=1]
        %6 = shl i32 %5, 24             ; <i32> [#uses=1]
        %7 = or i32 0, %6               ; <i32> [#uses=1]
        store i32 %7, ptr null, align 4
        br label %bb1
}
