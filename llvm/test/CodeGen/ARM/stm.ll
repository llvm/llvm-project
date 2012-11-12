; RUN: llc < %s -mtriple=arm-apple-darwin -mattr=+v6,+vfp2 | FileCheck %s

@"\01LC" = internal constant [32 x i8] c"Boolean Not: %d %d %d %d %d %d\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[32 x i8]*> [#uses=1]
@"\01LC1" = internal constant [26 x i8] c"Bitwise Not: %d %d %d %d\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[26 x i8]*> [#uses=1]

declare i32 @printf(i8* nocapture, ...) nounwind

define i32 @main() nounwind {
entry:
; CHECK: main
; CHECK: push
; CHECK: stm
	%0 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([26 x i8]* @"\01LC1", i32 0, i32 0), i32 -2, i32 -3, i32 2, i32 -6) nounwind		; <i32> [#uses=0]
	%1 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([32 x i8]* @"\01LC", i32 0, i32 0), i32 0, i32 1, i32 0, i32 1, i32 0, i32 1) nounwind		; <i32> [#uses=0]
	ret i32 0
}
