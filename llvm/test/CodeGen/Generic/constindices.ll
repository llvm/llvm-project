; RUN: llc < %s

; Test that a sequence of constant indices are folded correctly
; into the equivalent offset at compile-time.

        %MixedA = type { float, [15 x i32], i8, float }
        %MixedB = type { float, %MixedA, float }
@fmtArg = internal global [44 x i8] c"sqrt(2) = %g\0Aexp(1) = %g\0Api = %g\0Afive = %g\0A\00"           ; <ptr> [#uses=1]

declare i32 @printf(ptr, ...)

define i32 @main() {
        %ScalarA = alloca %MixedA               ; <ptr> [#uses=1]
        %ScalarB = alloca %MixedB               ; <ptr> [#uses=1]
        %ArrayA = alloca %MixedA, i32 4         ; <ptr> [#uses=3]
        %ArrayB = alloca %MixedB, i32 3         ; <ptr> [#uses=2]
        %I1 = getelementptr %MixedA, ptr %ScalarA, i64 0, i32 0             ; <ptr> [#uses=2]
        store float 0x3FF6A09020000000, ptr %I1
        %I2 = getelementptr %MixedB, ptr %ScalarB, i64 0, i32 1, i32 0              ; <ptr> [#uses=2]
        store float 0x4005BF1420000000, ptr %I2
        %fptrA = getelementptr %MixedA, ptr %ArrayA, i64 1, i32 0           ; <ptr> [#uses=1]
        %fptrB = getelementptr %MixedB, ptr %ArrayB, i64 2, i32 1, i32 0            ; <ptr> [#uses=1]
        store float 0x400921CAC0000000, ptr %fptrA
        store float 5.000000e+00, ptr %fptrB

        ;; Test that a sequence of GEPs with constant indices are folded right
        %fptrA1 = getelementptr %MixedA, ptr %ArrayA, i64 3         ; <ptr> [#uses=1]
        %fptrA2 = getelementptr %MixedA, ptr %fptrA1, i64 0, i32 1          ; <ptr> [#uses=1]
        %fptrA3 = getelementptr [15 x i32], ptr %fptrA2, i64 0, i64 8               ; <ptr> [#uses=1]
        store i32 5, ptr %fptrA3
        %sqrtTwo = load float, ptr %I1              ; <float> [#uses=1]
        %exp = load float, ptr %I2          ; <float> [#uses=1]
        %I3 = getelementptr %MixedA, ptr %ArrayA, i64 1, i32 0              ; <ptr> [#uses=1]
        %pi = load float, ptr %I3           ; <float> [#uses=1]
        %I4 = getelementptr %MixedB, ptr %ArrayB, i64 2, i32 1, i32 0               ; <ptr> [#uses=1]
        %five = load float, ptr %I4         ; <float> [#uses=1]
        %dsqrtTwo = fpext float %sqrtTwo to double              ; <double> [#uses=1]
        %dexp = fpext float %exp to double              ; <double> [#uses=1]
        %dpi = fpext float %pi to double                ; <double> [#uses=1]
        %dfive = fpext float %five to double            ; <double> [#uses=1]
        %castFmt = getelementptr [44 x i8], ptr @fmtArg, i64 0, i64 0               ; <ptr> [#uses=1]
        call i32 (ptr, ...) @printf( ptr %castFmt, double %dsqrtTwo, double %dexp, double %dpi, double %dfive )     ; <i32>:1 [#uses=0]
        ret i32 0
}
