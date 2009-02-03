;RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep zext

; Make sure the uint isn't removed.  Instcombine in llvm 1.9 was dropping the 
; uint cast which was causing a sign extend. This only affected code with 
; pointers in the high half of memory, so it wasn't noticed much
; compile a kernel though...

target datalayout = "e-p:32:32"
@str = internal constant [6 x i8] c"%llx\0A\00"         ; <[6 x i8]*> [#uses=1]

declare i32 @printf(i8*, ...)

define i32 @main(i32 %x, i8** %a) {
entry:
        %tmp = getelementptr [6 x i8]* @str, i32 0, i64 0               ; <i8*> [#uses=1]
        %tmp1 = load i8** %a            ; <i8*> [#uses=1]
        %tmp2 = ptrtoint i8* %tmp1 to i32               ; <i32> [#uses=1]
        %tmp3 = zext i32 %tmp2 to i64           ; <i64> [#uses=1]
        %tmp.upgrd.1 = call i32 (i8*, ...)* @printf( i8* %tmp, i64 %tmp3 )              ; <i32> [#uses=0]
        ret i32 0
}

