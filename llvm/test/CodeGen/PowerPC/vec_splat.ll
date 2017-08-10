; Test that vectors are scalarized/lowered correctly.
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g3 | \
; RUN:    grep stfs | count 4
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g5 -o %t
; RUN: grep vspltw %t | count 2
; RUN: grep vsplti %t | count 3
; RUN: grep vsplth %t | count 1

        %f4 = type <4 x float>
        %i4 = type <4 x i32>

define void @splat(%f4* %P, %f4* %Q, float %X) nounwind {
        %tmp = insertelement %f4 undef, float %X, i32 0         ; <%f4> [#uses=1]
        %tmp2 = insertelement %f4 %tmp, float %X, i32 1         ; <%f4> [#uses=1]
        %tmp4 = insertelement %f4 %tmp2, float %X, i32 2                ; <%f4> [#uses=1]
        %tmp6 = insertelement %f4 %tmp4, float %X, i32 3                ; <%f4> [#uses=1]
        %q = load %f4, %f4* %Q               ; <%f4> [#uses=1]
        %R = fadd %f4 %q, %tmp6          ; <%f4> [#uses=1]
        store %f4 %R, %f4* %P
        ret void
}

define void @splat_i4(%i4* %P, %i4* %Q, i32 %X) nounwind {
        %tmp = insertelement %i4 undef, i32 %X, i32 0           ; <%i4> [#uses=1]
        %tmp2 = insertelement %i4 %tmp, i32 %X, i32 1           ; <%i4> [#uses=1]
        %tmp4 = insertelement %i4 %tmp2, i32 %X, i32 2          ; <%i4> [#uses=1]
        %tmp6 = insertelement %i4 %tmp4, i32 %X, i32 3          ; <%i4> [#uses=1]
        %q = load %i4, %i4* %Q               ; <%i4> [#uses=1]
        %R = add %i4 %q, %tmp6          ; <%i4> [#uses=1]
        store %i4 %R, %i4* %P
        ret void
}

define void @splat_imm_i32(%i4* %P, %i4* %Q, i32 %X) nounwind {
        %q = load %i4, %i4* %Q               ; <%i4> [#uses=1]
        %R = add %i4 %q, < i32 -1, i32 -1, i32 -1, i32 -1 >             ; <%i4> [#uses=1]
        store %i4 %R, %i4* %P
        ret void
}

define void @splat_imm_i16(%i4* %P, %i4* %Q, i32 %X) nounwind {
        %q = load %i4, %i4* %Q               ; <%i4> [#uses=1]
        %R = add %i4 %q, < i32 65537, i32 65537, i32 65537, i32 65537 >         ; <%i4> [#uses=1]
        store %i4 %R, %i4* %P
        ret void
}

define void @splat_h(i16 %tmp, <16 x i8>* %dst) nounwind {
        %tmp.upgrd.1 = insertelement <8 x i16> undef, i16 %tmp, i32 0           
        %tmp72 = insertelement <8 x i16> %tmp.upgrd.1, i16 %tmp, i32 1 
        %tmp73 = insertelement <8 x i16> %tmp72, i16 %tmp, i32 2 
        %tmp74 = insertelement <8 x i16> %tmp73, i16 %tmp, i32 3
        %tmp75 = insertelement <8 x i16> %tmp74, i16 %tmp, i32 4 
        %tmp76 = insertelement <8 x i16> %tmp75, i16 %tmp, i32 5
        %tmp77 = insertelement <8 x i16> %tmp76, i16 %tmp, i32 6 
        %tmp78 = insertelement <8 x i16> %tmp77, i16 %tmp, i32 7 
        %tmp78.upgrd.2 = bitcast <8 x i16> %tmp78 to <16 x i8>  
        store <16 x i8> %tmp78.upgrd.2, <16 x i8>* %dst
        ret void
}

define void @spltish(<16 x i8>* %A, <16 x i8>* %B) nounwind {
        %tmp = load <16 x i8>, <16 x i8>* %B               ; <<16 x i8>> [#uses=1]
        %tmp.s = bitcast <16 x i8> %tmp to <16 x i8>            ; <<16 x i8>> [#uses=1]
        %tmp4 = sub <16 x i8> %tmp.s, bitcast (<8 x i16> < i16 15, i16 15, i16 15, i16 15, i16 15, i16
 15, i16 15, i16 15 > to <16 x i8>)             ; <<16 x i8>> [#uses=1]
        %tmp4.u = bitcast <16 x i8> %tmp4 to <16 x i8>          ; <<16 x i8>> [#uses=1]
        store <16 x i8> %tmp4.u, <16 x i8>* %A
        ret void
}

