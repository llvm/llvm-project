; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   grep {sub float}
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   grep {mul float}
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   not grep {insertelement.*0.00}
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   not grep {call.*llvm.x86.sse.mul}
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   not grep {call.*llvm.x86.sse.sub}
; END.

define i16 @test1(float %f) {
entry:
	%tmp = insertelement <4 x float> undef, float %f, i32 0		; <<4 x float>> [#uses=1]
	%tmp10 = insertelement <4 x float> %tmp, float 0.000000e+00, i32 1		; <<4 x float>> [#uses=1]
	%tmp11 = insertelement <4 x float> %tmp10, float 0.000000e+00, i32 2		; <<4 x float>> [#uses=1]
	%tmp12 = insertelement <4 x float> %tmp11, float 0.000000e+00, i32 3		; <<4 x float>> [#uses=1]
	%tmp28 = tail call <4 x float> @llvm.x86.sse.sub.ss( <4 x float> %tmp12, <4 x float> < float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp37 = tail call <4 x float> @llvm.x86.sse.mul.ss( <4 x float> %tmp28, <4 x float> < float 5.000000e-01, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp48 = tail call <4 x float> @llvm.x86.sse.min.ss( <4 x float> %tmp37, <4 x float> < float 6.553500e+04, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp59 = tail call <4 x float> @llvm.x86.sse.max.ss( <4 x float> %tmp48, <4 x float> zeroinitializer )		; <<4 x float>> [#uses=1]
	%tmp.upgrd.1 = tail call i32 @llvm.x86.sse.cvttss2si( <4 x float> %tmp59 )		; <i32> [#uses=1]
	%tmp69 = trunc i32 %tmp.upgrd.1 to i16		; <i16> [#uses=1]
	ret i16 %tmp69
}

define i32 @test2(float %f) {
        %tmp5 = mul float %f, %f
        %tmp9 = insertelement <4 x float> undef, float %tmp5, i32 0             
        %tmp10 = insertelement <4 x float> %tmp9, float 0.000000e+00, i32 1    
        %tmp11 = insertelement <4 x float> %tmp10, float 0.000000e+00, i32 2  
        %tmp12 = insertelement <4 x float> %tmp11, float 0.000000e+00, i32 3 
        %tmp19 = bitcast <4 x float> %tmp12 to <4 x i32>  
        %tmp21 = extractelement <4 x i32> %tmp19, i32 0  
        ret i32 %tmp21
}

declare <4 x float> @llvm.x86.sse.sub.ss(<4 x float>, <4 x float>)

declare <4 x float> @llvm.x86.sse.mul.ss(<4 x float>, <4 x float>)

declare <4 x float> @llvm.x86.sse.min.ss(<4 x float>, <4 x float>)

declare <4 x float> @llvm.x86.sse.max.ss(<4 x float>, <4 x float>)

declare i32 @llvm.x86.sse.cvttss2si(<4 x float>)
