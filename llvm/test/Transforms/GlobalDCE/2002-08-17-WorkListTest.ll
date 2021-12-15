; This testcase tests that a worklist is being used, and that globals can be 
; removed if they are the subject of a constexpr and ConstantPointerRef

; RUN: opt < %s -passes=globaldce -S | FileCheck %s

; CHECK-NOT: global

@t0 = internal global [4 x i8] c"foo\00"                ; <[4 x i8]*> [#uses=1]
@t1 = internal global [4 x i8] c"bar\00"                ; <[4 x i8]*> [#uses=1]
@s1 = internal global [1 x i8*] [ i8* getelementptr ([4 x i8], [4 x i8]* @t0, i32 0, i32 0) ]             ; <[1 x i8*]*> [#uses=0]
@s2 = internal global [1 x i8*] [ i8* getelementptr ([4 x i8], [4 x i8]* @t1, i64 0, i64 0) ]             ; <[1 x i8*]*> [#uses=0]
@b = internal global i32* @a            ; <i32**> [#uses=0]
@a = internal global i32 7              ; <i32*> [#uses=1]

