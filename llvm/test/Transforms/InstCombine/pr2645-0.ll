; RUN: opt < %s -instcombine -S | grep "insertelement <4 x float> undef"

; Instcombine should be able to prove that none of the
; insertelement's first operand's elements are needed.

define internal void @""(i8*) {
; <label>:1
        bitcast i8* %0 to i32*          ; <i32*>:2 [#uses=1]
        load i32, i32* %2, align 1           ; <i32>:3 [#uses=1]
        getelementptr i8, i8* %0, i32 4             ; <i8*>:4 [#uses=1]
        bitcast i8* %4 to i32*          ; <i32*>:5 [#uses=1]
        load i32, i32* %5, align 1           ; <i32>:6 [#uses=1]
        br label %7

; <label>:7             ; preds = %9, %1
        %.01 = phi <4 x float> [ undef, %1 ], [ %12, %9 ]               ; <<4 x float>> [#uses=1]
        %.0 = phi i32 [ %3, %1 ], [ %15, %9 ]           ; <i32> [#uses=3]
        icmp slt i32 %.0, %6            ; <i1>:8 [#uses=1]
        br i1 %8, label %9, label %16

; <label>:9             ; preds = %7
        sitofp i32 %.0 to float         ; <float>:10 [#uses=1]
        insertelement <4 x float> %.01, float %10, i32 0                ; <<4 x float>>:11 [#uses=1]
        shufflevector <4 x float> %11, <4 x float> undef, <4 x i32> zeroinitializer             ; <<4 x float>>:12 [#uses=2]
        getelementptr i8, i8* %0, i32 48            ; <i8*>:13 [#uses=1]
        bitcast i8* %13 to <4 x float>*         ; <<4 x float>*>:14 [#uses=1]
        store <4 x float> %12, <4 x float>* %14, align 16
        add i32 %.0, 2          ; <i32>:15 [#uses=1]
        br label %7

; <label>:16            ; preds = %7
        ret void
}
