; RUN: opt < %s -simplify-libcalls -S | FileCheck %s

@str = internal constant [13 x i8] c"hello world\0A\00"         ; <[13 x i8]*> [#uses=1]
@str1 = internal constant [2 x i8] c"h\00"              ; <[2 x i8]*> [#uses=1]

; CHECK: private unnamed_addr constant [12 x i8] c"hello world\00"

declare i32 @printf(i8*, ...)

; CHECK: define void @f0
; CHECK-NOT: printf
; CHECK: }
define void @f0() {
entry:
        %tmp1 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([13 x i8]* @str, i32 0, i32 0) )         ; <i32> [#uses=0]
        ret void
}

; CHECK: define void @f1
; CHECK-NOT: printf
; CHECK: }
define void @f1() {
entry:
        %tmp1 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([2 x i8]* @str1, i32 0, i32 0) )         ; <i32> [#uses=0]
        ret void
}

; Verify that we don't turn this into a putchar call (thus changing the return
; value).
;
; CHECK: define i32 @f2
; CHECK: printf
; CHECK: }
define i32 @f2() {
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([2 x i8]* @str1, i32 0, i32 0))
  ret i32 %call
}
