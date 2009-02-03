; Tests to make sure elimination of casts is working correctly
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep %c | notcast

@inbuf = external global [32832 x i8]           ; <[32832 x i8]*> [#uses=1]

define i32 @test1(i32 %A) {
        %c1 = bitcast i32 %A to i32             ; <i32> [#uses=1]
        %c2 = bitcast i32 %c1 to i32            ; <i32> [#uses=1]
        ret i32 %c2
}

define i64 @test2(i8 %A) {
        %c1 = zext i8 %A to i16         ; <i16> [#uses=1]
        %c2 = zext i16 %c1 to i32               ; <i32> [#uses=1]
        %Ret = zext i32 %c2 to i64              ; <i64> [#uses=1]
        ret i64 %Ret
}

; This function should just use bitwise AND
define i64 @test3(i64 %A) {
        %c1 = trunc i64 %A to i8                ; <i8> [#uses=1]
        %c2 = zext i8 %c1 to i64                ; <i64> [#uses=1]
        ret i64 %c2
}

define i32 @test4(i32 %A, i32 %B) {
        %COND = icmp slt i32 %A, %B             ; <i1> [#uses=1]
        ; Booleans are unsigned integrals
        %c = zext i1 %COND to i8                ; <i8> [#uses=1]
        ; for the cast elim purpose
        %result = zext i8 %c to i32             ; <i32> [#uses=1]
        ret i32 %result
}

define i32 @test5(i1 %B) {
        ; This cast should get folded into
        %c = zext i1 %B to i8           ; <i8> [#uses=1]
        ; this cast        
        %result = zext i8 %c to i32             ; <i32> [#uses=1]
        ret i32 %result
}

define i32 @test6(i64 %A) {
        %c1 = trunc i64 %A to i32               ; <i32> [#uses=1]
        %res = bitcast i32 %c1 to i32           ; <i32> [#uses=1]
        ret i32 %res
}

define i64 @test7(i1 %A) {
        %c1 = zext i1 %A to i32         ; <i32> [#uses=1]
        %res = sext i32 %c1 to i64              ; <i64> [#uses=1]
        ret i64 %res
}

define i64 @test8(i8 %A) {
        %c1 = sext i8 %A to i64         ; <i64> [#uses=1]
        %res = bitcast i64 %c1 to i64           ; <i64> [#uses=1]
        ret i64 %res
}

define i16 @test9(i16 %A) {
        %c1 = sext i16 %A to i32                ; <i32> [#uses=1]
        %c2 = trunc i32 %c1 to i16              ; <i16> [#uses=1]
        ret i16 %c2
}

define i16 @test10(i16 %A) {
        %c1 = sext i16 %A to i32                ; <i32> [#uses=1]
        %c2 = trunc i32 %c1 to i16              ; <i16> [#uses=1]
        ret i16 %c2
}

declare void @varargs(i32, ...)

define void @test11(i32* %P) {
        %c = bitcast i32* %P to i16*            ; <i16*> [#uses=1]
        call void (i32, ...)* @varargs( i32 5, i16* %c )
        ret void
}

define i32* @test12() {
        %p = malloc [4 x i8]            ; <[4 x i8]*> [#uses=1]
        %c = bitcast [4 x i8]* %p to i32*               ; <i32*> [#uses=1]
        ret i32* %c
}
define i8* @test13(i64 %A) {
        %c = getelementptr [0 x i8]* bitcast ([32832 x i8]* @inbuf to [0 x i8]*), i64 0, i64 %A             ; <i8*> [#uses=1]
        ret i8* %c
}

define i1 @test14(i8 %A) {
        %c = bitcast i8 %A to i8                ; <i8> [#uses=1]
        %X = icmp ult i8 %c, -128               ; <i1> [#uses=1]
        ret i1 %X
}


; This just won't occur when there's no difference between ubyte and sbyte
;bool %test15(ubyte %A) {
;        %c = cast ubyte %A to sbyte
;        %X = setlt sbyte %c, 0   ; setgt %A, 127
;        ret bool %X
;}

define i1 @test16(i32* %P) {
        %c = icmp ne i32* %P, null              ; <i1> [#uses=1]
        ret i1 %c
}

define i16 @test17(i1 %tmp3) {
        %c = zext i1 %tmp3 to i32               ; <i32> [#uses=1]
        %t86 = trunc i32 %c to i16              ; <i16> [#uses=1]
        ret i16 %t86
}

define i16 @test18(i8 %tmp3) {
        %c = sext i8 %tmp3 to i32               ; <i32> [#uses=1]
        %t86 = trunc i32 %c to i16              ; <i16> [#uses=1]
        ret i16 %t86
}

define i1 @test19(i32 %X) {
        %c = sext i32 %X to i64         ; <i64> [#uses=1]
        %Z = icmp slt i64 %c, 12345             ; <i1> [#uses=1]
        ret i1 %Z
}

define i1 @test20(i1 %B) {
        %c = zext i1 %B to i32          ; <i32> [#uses=1]
        %D = icmp slt i32 %c, -1                ; <i1> [#uses=1]
        ;; false
        ret i1 %D
}

define i32 @test21(i32 %X) {
        %c1 = trunc i32 %X to i8                ; <i8> [#uses=1]
        ;; sext -> zext -> and -> nop
        %c2 = sext i8 %c1 to i32                ; <i32> [#uses=1]
        %RV = and i32 %c2, 255          ; <i32> [#uses=1]
        ret i32 %RV
}

define i32 @test22(i32 %X) {
        %c1 = trunc i32 %X to i8                ; <i8> [#uses=1]
        ;; sext -> zext -> and -> nop
        %c2 = sext i8 %c1 to i32                ; <i32> [#uses=1]
        %RV = shl i32 %c2, 24           ; <i32> [#uses=1]
        ret i32 %RV
}

define i32 @test23(i32 %X) {
        ;; Turn into an AND even though X
        %c1 = trunc i32 %X to i16               ; <i16> [#uses=1]
        ;; and Z are signed.
        %c2 = zext i16 %c1 to i32               ; <i32> [#uses=1]
        ret i32 %c2
}

define i1 @test24(i1 %C) {
        %X = select i1 %C, i32 14, i32 1234             ; <i32> [#uses=1]
        ;; Fold cast into select
        %c = icmp ne i32 %X, 0          ; <i1> [#uses=1]
        ret i1 %c
}

define void @test25(i32** %P) {
        %c = bitcast i32** %P to float**                ; <float**> [#uses=1]
        ;; Fold cast into null
        store float* null, float** %c
        ret void
}

define i32 @test26(float %F) {
        ;; no need to cast from float->double.
        %c = fpext float %F to double           ; <double> [#uses=1]
        %D = fptosi double %c to i32            ; <i32> [#uses=1]
        ret i32 %D
}

define [4 x float]* @test27([9 x [4 x float]]* %A) {
        %c = bitcast [9 x [4 x float]]* %A to [4 x float]*              ; <[4 x float]*> [#uses=1]
        ret [4 x float]* %c
}

define float* @test28([4 x float]* %A) {
        %c = bitcast [4 x float]* %A to float*          ; <float*> [#uses=1]
        ret float* %c
}

define i32 @test29(i32 %c1, i32 %c2) {
        %tmp1 = trunc i32 %c1 to i8             ; <i8> [#uses=1]
        %tmp4.mask = trunc i32 %c2 to i8                ; <i8> [#uses=1]
        %tmp = or i8 %tmp4.mask, %tmp1          ; <i8> [#uses=1]
        %tmp10 = zext i8 %tmp to i32            ; <i32> [#uses=1]
        ret i32 %tmp10
}

define i32 @test30(i32 %c1) {
        %c2 = trunc i32 %c1 to i8               ; <i8> [#uses=1]
        %c3 = xor i8 %c2, 1             ; <i8> [#uses=1]
        %c4 = zext i8 %c3 to i32                ; <i32> [#uses=1]
        ret i32 %c4
}

define i1 @test31(i64 %A) {
        %B = trunc i64 %A to i32                ; <i32> [#uses=1]
        %C = and i32 %B, 42             ; <i32> [#uses=1]
        %D = icmp eq i32 %C, 10         ; <i1> [#uses=1]
        ret i1 %D
}

define void @test32(double** %tmp) {
        %tmp8 = malloc [16 x i8]                ; <[16 x i8]*> [#uses=1]
        %tmp8.upgrd.1 = bitcast [16 x i8]* %tmp8 to double*             ; <double*> [#uses=1]
        store double* %tmp8.upgrd.1, double** %tmp
        ret void
}

define i32 @test33(i32 %c1) {
        %x = bitcast i32 %c1 to float           ; <float> [#uses=1]
        %y = bitcast float %x to i32            ; <i32> [#uses=1]
        ret i32 %y
}

define i16 @test34(i16 %a) {
        %c1 = zext i16 %a to i32                ; <i32> [#uses=1]
        %tmp21 = lshr i32 %c1, 8                ; <i32> [#uses=1]
        %c2 = trunc i32 %tmp21 to i16           ; <i16> [#uses=1]
        ret i16 %c2
}

define i16 @test35(i16 %a) {
        %c1 = bitcast i16 %a to i16             ; <i16> [#uses=1]
        %tmp2 = lshr i16 %c1, 8         ; <i16> [#uses=1]
        %c2 = bitcast i16 %tmp2 to i16          ; <i16> [#uses=1]
        ret i16 %c2
}

; icmp sgt i32 %a, -1
; rdar://6480391
define i1 @test36(i32 %a) {
        %b = lshr i32 %a, 31
        %c = trunc i32 %b to i8
        %d = icmp eq i8 %c, 0
        ret i1 %d
}

; ret i1 false
define i1 @test37(i32 %a) {
        %b = lshr i32 %a, 31
        %c = or i32 %b, 512
        %d = trunc i32 %c to i8
        %e = icmp eq i8 %d, 11
        ret i1 %e
}

define i64 @test38(i32 %a) {
	%1 = icmp eq i32 %a, -2
	%2 = zext i1 %1 to i8
	%3 = xor i8 %2, 1
	%4 = zext i8 %3 to i64
        ret i64 %4
}
