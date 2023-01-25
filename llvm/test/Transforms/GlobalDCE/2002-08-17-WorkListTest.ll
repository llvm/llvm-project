; This testcase tests that a worklist is being used, and that globals can be 
; removed if they are the subject of a constexpr and ConstantPointerRef

; RUN: opt < %s -passes=globaldce -S | FileCheck %s

; CHECK-NOT: global

@t0 = internal global [4 x i8] c"foo\00"                ; <ptr> [#uses=1]
@t1 = internal global [4 x i8] c"bar\00"                ; <ptr> [#uses=1]
@s1 = internal global [1 x ptr] [ ptr @t0 ]             ; <ptr> [#uses=0]
@s2 = internal global [1 x ptr] [ ptr @t1 ]             ; <ptr> [#uses=0]
@b = internal global ptr @a            ; <ptr> [#uses=0]
@a = internal global i32 7              ; <ptr> [#uses=1]

