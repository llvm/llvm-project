; RUN: opt < %s -instcombine -S | grep shufflevector
; PR2645

; instcombine shouldn't delete the shufflevector.

define internal void @""(i8*, i32, i8*) {
; <label>:3
        br label %4

; <label>:4             ; preds = %6, %3
        %.0 = phi i32 [ 0, %3 ], [ %19, %6 ]            ; <i32> [#uses=4]
        %5 = icmp slt i32 %.0, %1               ; <i1> [#uses=1]
        br i1 %5, label %6, label %20

; <label>:6             ; preds = %4
        %7 = getelementptr i8, i8* %2, i32 %.0              ; <i8*> [#uses=1]
        %8 = bitcast i8* %7 to <4 x i16>*               ; <<4 x i16>*> [#uses=1]
        %9 = load <4 x i16>, <4 x i16>* %8, align 1                ; <<4 x i16>> [#uses=1]
        %10 = bitcast <4 x i16> %9 to <1 x i64>         ; <<1 x i64>> [#uses=1]
        %11 = call <2 x i64> @foo(<1 x i64> %10)
; <<2 x i64>> [#uses=1]
        %12 = bitcast <2 x i64> %11 to <4 x i32>                ; <<4 x i32>> [#uses=1]
        %13 = bitcast <4 x i32> %12 to <8 x i16>                ; <<8 x i16>> [#uses=2]
        %14 = shufflevector <8 x i16> %13, <8 x i16> %13, <8 x i32> < i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3 >          ; <<8 x i16>> [#uses=1]
        %15 = bitcast <8 x i16> %14 to <4 x i32>                ; <<4 x i32>> [#uses=1]
        %16 = sitofp <4 x i32> %15 to <4 x float>               ; <<4 x float>> [#uses=1]
        %17 = getelementptr i8, i8* %0, i32 %.0             ; <i8*> [#uses=1]
        %18 = bitcast i8* %17 to <4 x float>*           ; <<4 x float>*> [#uses=1]
        store <4 x float> %16, <4 x float>* %18, align 1
        %19 = add i32 %.0, 1            ; <i32> [#uses=1]
        br label %4

; <label>:20            ; preds = %4
        call void @llvm.x86.mmx.emms( )
        ret void
}

declare <2 x i64> @foo(<1 x i64>)
declare void @llvm.x86.mmx.emms( )
