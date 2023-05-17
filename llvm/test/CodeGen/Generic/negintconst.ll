; RUN: llc < %s

; Test that a negative constant smaller than 64 bits (e.g., int)
; is correctly implemented with sign-extension.
; In particular, the current code generated is:
;
; main:
; .L_main_LL_0:
;         save    %o6, -224, %o6
;         setx    .G_fmtArg_1, %o1, %o0
;         setuw   1, %o1		! i = 1
;         setuw   4294967295, %o3	! THE BUG: 0x00000000ffffffff
;         setsw   0, %i0
;         add     %i6, 1999, %o2	! fval
;         add     %o1, %g0, %o1
;         add     %o0, 0, %o0
;         mulx    %o1, %o3, %o1		! ERROR: 0xffffffff; should be -1
;         add     %o1, 3, %o1		! ERROR: 0x100000002; should be 0x2
;         mulx    %o1, 12, %o3		! 
;         add     %o2, %o3, %o3		! produces bad address!
;         call    printf
;         nop     
;         jmpl    %i7+8, %g0
;         restore %g0, 0, %g0
; 
;   llc produces:
; ioff = 2        fval = 0xffffffff7fffec90       &fval[2] = 0xb7fffeca8
;   instead of:
; ioff = 2        fval = 0xffffffff7fffec90       &fval[2] = 0xffffffff7fffeca8
; 
        %Results = type { float, float, float }
@fmtArg = internal global [39 x i8] c"ioff = %u\09fval = 0x%p\09&fval[2] = 0x%p\0A\00"          ; <ptr> [#uses=1]

declare i32 @printf(ptr, ...)

define i32 @main() {
        %fval = alloca %Results, i32 4          ; <ptr> [#uses=2]
        %i = add i32 1, 0               ; <i32> [#uses=1]
        %iscale = mul i32 %i, -1                ; <i32> [#uses=1]
        %ioff = add i32 %iscale, 3              ; <i32> [#uses=2]
        %ioff.upgrd.1 = zext i32 %ioff to i64           ; <i64> [#uses=1]
        %fptr = getelementptr %Results, ptr %fval, i64 %ioff.upgrd.1                ; <ptr> [#uses=1]
        %castFmt = getelementptr [39 x i8], ptr @fmtArg, i64 0, i64 0               ; <ptr> [#uses=1]
        call i32 (ptr, ...) @printf( ptr %castFmt, i32 %ioff, ptr %fval, ptr %fptr )               ; <i32>:1 [#uses=0]
        ret i32 0
}

