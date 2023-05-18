;RUN: opt < %s -passes=instcombine -S | grep zext

; Make sure the uint isn't removed.  Instcombine in llvm 1.9 was dropping the 
; uint cast which was causing a sign extend. This only affected code with 
; pointers in the high half of memory, so it wasn't noticed much
; compile a kernel though...

target datalayout = "e-p:32:32"
@str = internal constant [6 x i8] c"%llx\0A\00"         ; <ptr> [#uses=1]

declare i32 @printf(ptr, ...)

define i32 @main(i32 %x, ptr %a) {
entry:
        %tmp = getelementptr [6 x i8], ptr @str, i32 0, i64 0               ; <ptr> [#uses=1]
        %tmp1 = load ptr, ptr %a            ; <ptr> [#uses=1]
        %tmp2 = ptrtoint ptr %tmp1 to i32               ; <i32> [#uses=1]
        %tmp3 = zext i32 %tmp2 to i64           ; <i64> [#uses=1]
        %tmp.upgrd.1 = call i32 (ptr, ...) @printf( ptr %tmp, i64 %tmp3 )              ; <i32> [#uses=0]
        ret i32 0
}

