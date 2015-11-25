; RUN: opt < %s -instcombine -S | FileCheck %s

define <4 x float> @test1(<4 x float> %v1) {
; CHECK-LABEL: @test1(
; CHECK: ret <4 x float> %v1
  %v2 = shufflevector <4 x float> %v1, <4 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x float> %v2
}

define <4 x float> @test2(<4 x float> %v1) {
; CHECK-LABEL: @test2(
; CHECK: ret <4 x float> %v1
  %v2 = shufflevector <4 x float> %v1, <4 x float> %v1, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x float> %v2
}

define float @test3(<4 x float> %A, <4 x float> %B, float %f) {
; CHECK-LABEL: @test3(
; CHECK: ret float %f
        %C = insertelement <4 x float> %A, float %f, i32 0
        %D = shufflevector <4 x float> %C, <4 x float> %B, <4 x i32> <i32 5, i32 0, i32 2, i32 7>
        %E = extractelement <4 x float> %D, i32 1
        ret float %E
}

define i32 @test4(<4 x i32> %X) {
; CHECK-LABEL: @test4(
; CHECK-NEXT: extractelement
; CHECK-NEXT: ret 
        %tmp152.i53899.i = shufflevector <4 x i32> %X, <4 x i32> undef, <4 x i32> zeroinitializer
        %tmp34 = extractelement <4 x i32> %tmp152.i53899.i, i32 0
        ret i32 %tmp34
}

define i32 @test5(<4 x i32> %X) {
; CHECK-LABEL: @test5(
; CHECK-NEXT: extractelement
; CHECK-NEXT: ret 
        %tmp152.i53899.i = shufflevector <4 x i32> %X, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 undef, i32 undef>
        %tmp34 = extractelement <4 x i32> %tmp152.i53899.i, i32 0
        ret i32 %tmp34
}

define float @test6(<4 x float> %X) {
; CHECK-LABEL: @test6(
; CHECK-NEXT: extractelement
; CHECK-NEXT: ret 
        %X1 = bitcast <4 x float> %X to <4 x i32>
        %tmp152.i53899.i = shufflevector <4 x i32> %X1, <4 x i32> undef, <4 x i32> zeroinitializer
        %tmp152.i53900.i = bitcast <4 x i32> %tmp152.i53899.i to <4 x float>
        %tmp34 = extractelement <4 x float> %tmp152.i53900.i, i32 0
        ret float %tmp34
}

define <4 x float> @test7(<4 x float> %tmp45.i) {
; CHECK-LABEL: @test7(
; CHECK-NEXT: ret <4 x float> %tmp45.i
        %tmp1642.i = shufflevector <4 x float> %tmp45.i, <4 x float> undef, <4 x i32> < i32 0, i32 1, i32 6, i32 7 >
        ret <4 x float> %tmp1642.i
}

; This should turn into a single shuffle.
define <4 x float> @test8(<4 x float> %tmp, <4 x float> %tmp1) {
; CHECK-LABEL: @test8(
; CHECK-NEXT: shufflevector
; CHECK-NEXT: ret
        %tmp4 = extractelement <4 x float> %tmp, i32 1
        %tmp2 = extractelement <4 x float> %tmp, i32 3
        %tmp1.upgrd.1 = extractelement <4 x float> %tmp1, i32 0
        %tmp128 = insertelement <4 x float> undef, float %tmp4, i32 0
        %tmp130 = insertelement <4 x float> %tmp128, float undef, i32 1
        %tmp132 = insertelement <4 x float> %tmp130, float %tmp2, i32 2 
        %tmp134 = insertelement <4 x float> %tmp132, float %tmp1.upgrd.1, i32 3
        ret <4 x float> %tmp134
}

; Test fold of two shuffles where the first shuffle vectors inputs are a
; different length then the second.
define <4 x i8> @test9(<16 x i8> %tmp6) nounwind {
; CHECK-LABEL: @test9(
; CHECK-NEXT: shufflevector
; CHECK-NEXT: ret
	%tmp7 = shufflevector <16 x i8> %tmp6, <16 x i8> undef, <4 x i32> < i32 13, i32 9, i32 4, i32 13 >		; <<4 x i8>> [#uses=1]
	%tmp9 = shufflevector <4 x i8> %tmp7, <4 x i8> undef, <4 x i32> < i32 3, i32 1, i32 2, i32 0 >		; <<4 x i8>> [#uses=1]
	ret <4 x i8> %tmp9
}

; Same as test9, but make sure that "undef" mask values are not confused with
; mask values of 2*N, where N is the mask length.  These shuffles should not
; be folded (because [8,9,4,8] may not be a mask supported by the target).
define <4 x i8> @test9a(<16 x i8> %tmp6) nounwind {
; CHECK-LABEL: @test9a(
; CHECK-NEXT: shufflevector
; CHECK-NEXT: shufflevector
; CHECK-NEXT: ret
	%tmp7 = shufflevector <16 x i8> %tmp6, <16 x i8> undef, <4 x i32> < i32 undef, i32 9, i32 4, i32 8 >		; <<4 x i8>> [#uses=1]
	%tmp9 = shufflevector <4 x i8> %tmp7, <4 x i8> undef, <4 x i32> < i32 3, i32 1, i32 2, i32 0 >		; <<4 x i8>> [#uses=1]
	ret <4 x i8> %tmp9
}

; Test fold of two shuffles where the first shuffle vectors inputs are a
; different length then the second.
define <4 x i8> @test9b(<4 x i8> %tmp6, <4 x i8> %tmp7) nounwind {
; CHECK-LABEL: @test9b(
; CHECK-NEXT: shufflevector
; CHECK-NEXT: ret
  %tmp1 = shufflevector <4 x i8> %tmp6, <4 x i8> %tmp7, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 4, i32 5, i32 2, i32 3>		; <<4 x i8>> [#uses=1]
  %tmp9 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <4 x i32> <i32 0, i32 1, i32 4, i32 5>		; <<4 x i8>> [#uses=1]
  ret <4 x i8> %tmp9
}

; Redundant vector splats should be removed.  Radar 8597790.
define <4 x i32> @test10(<4 x i32> %tmp5) nounwind {
; CHECK-LABEL: @test10(
; CHECK-NEXT: shufflevector
; CHECK-NEXT: ret
  %tmp6 = shufflevector <4 x i32> %tmp5, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %tmp7 = shufflevector <4 x i32> %tmp6, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %tmp7
}

; Test fold of two shuffles where the two shufflevector inputs's op1 are
; the same
define <8 x i8> @test11(<16 x i8> %tmp6) nounwind {
; CHECK-LABEL: @test11(
; CHECK-NEXT: shufflevector <16 x i8> %tmp6, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT: ret
  %tmp1 = shufflevector <16 x i8> %tmp6, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>		; <<4 x i8>> [#uses=1]
  %tmp2 = shufflevector <16 x i8> %tmp6, <16 x i8> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>		; <<4 x i8>> [#uses=1]
  %tmp3 = shufflevector <4 x i8> %tmp1, <4 x i8> %tmp2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>		; <<8 x i8>> [#uses=1]
  ret <8 x i8> %tmp3
}

; Test fold of two shuffles where the first shufflevector's inputs are
; the same as the second
define <8 x i8> @test12(<8 x i8> %tmp6, <8 x i8> %tmp2) nounwind {
; CHECK-LABEL: @test12(
; CHECK-NEXT: shufflevector <8 x i8> %tmp6, <8 x i8> %tmp2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 9, i32 8, i32 11, i32 12>
; CHECK-NEXT: ret
  %tmp1 = shufflevector <8 x i8> %tmp6, <8 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 undef, i32 7>	; <<8 x i8>> [#uses=1]
  %tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 9, i32 8, i32 11, i32 12>		; <<8 x i8>> [#uses=1]
  ret <8 x i8> %tmp3
}

; Test fold of two shuffles where the first shufflevector's inputs are
; the same as the second
define <8 x i8> @test12a(<8 x i8> %tmp6, <8 x i8> %tmp2) nounwind {
; CHECK-LABEL: @test12a(
; CHECK-NEXT: shufflevector <8 x i8> %tmp2, <8 x i8> %tmp6, <8 x i32> <i32 0, i32 3, i32 1, i32 4, i32 8, i32 9, i32 10, i32 11>
; CHECK-NEXT: ret
  %tmp1 = shufflevector <8 x i8> %tmp6, <8 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 undef, i32 7>	; <<8 x i8>> [#uses=1]
  %tmp3 = shufflevector <8 x i8> %tmp2, <8 x i8> %tmp1, <8 x i32> <i32 0, i32 3, i32 1, i32 4, i32 8, i32 9, i32 10, i32 11>		; <<8 x i8>> [#uses=1]
  ret <8 x i8> %tmp3
}

define <2 x i8> @test13a(i8 %x1, i8 %x2) {
; CHECK-LABEL: @test13a(
; CHECK-NEXT: insertelement {{.*}} undef, i8 %x1, i32 1
; CHECK-NEXT: insertelement {{.*}} i8 %x2, i32 0
; CHECK-NEXT: add {{.*}} <i8 7, i8 5>
; CHECK-NEXT: ret
  %A = insertelement <2 x i8> undef, i8 %x1, i32 0
  %B = insertelement <2 x i8> %A, i8 %x2, i32 1
  %C = add <2 x i8> %B, <i8 5, i8 7>
  %D = shufflevector <2 x i8> %C, <2 x i8> undef, <2 x i32> <i32 1, i32 0>
  ret <2 x i8> %D
}

define <2 x i8> @test13b(i8 %x) {
; CHECK-LABEL: @test13b(
; CHECK-NEXT: insertelement <2 x i8> undef, i8 %x, i32 1
; CHECK-NEXT: ret
  %A = insertelement <2 x i8> undef, i8 %x, i32 0
  %B = shufflevector <2 x i8> %A, <2 x i8> undef, <2 x i32> <i32 undef, i32 0>
  ret <2 x i8> %B
}

define <2 x i8> @test13c(i8 %x1, i8 %x2) {
; CHECK-LABEL: @test13c(
; CHECK-NEXT: insertelement <2 x i8> {{.*}}, i32 0
; CHECK-NEXT: insertelement <2 x i8> {{.*}}, i32 1
; CHECK-NEXT: ret
  %A = insertelement <4 x i8> undef, i8 %x1, i32 0
  %B = insertelement <4 x i8> %A, i8 %x2, i32 2
  %C = shufflevector <4 x i8> %B, <4 x i8> undef, <2 x i32> <i32 0, i32 2>
  ret <2 x i8> %C
}

define void @test14(i16 %conv10) {
  %tmp = alloca <4 x i16>, align 8
  %vecinit6 = insertelement <4 x i16> undef, i16 23, i32 3
  store <4 x i16> %vecinit6, <4 x i16>* undef
  %tmp1 = load <4 x i16>, <4 x i16>* undef
  %vecinit11 = insertelement <4 x i16> undef, i16 %conv10, i32 3
  %div = udiv <4 x i16> %tmp1, %vecinit11
  store <4 x i16> %div, <4 x i16>* %tmp
  %tmp4 = load <4 x i16>, <4 x i16>* %tmp
  %tmp5 = shufflevector <4 x i16> %tmp4, <4 x i16> undef, <2 x i32> <i32 2, i32 0>
  %cmp = icmp ule <2 x i16> %tmp5, undef
  %sext = sext <2 x i1> %cmp to <2 x i16>
  ret void
}

; Check that sequences of insert/extract element are 
; collapsed into valid shuffle instruction with correct shuffle indexes.
 
define <4 x float> @test15a(<4 x float> %LHS, <4 x float> %RHS) {
; CHECK-LABEL: @test15a
; CHECK-NEXT: shufflevector <4 x float> %LHS, <4 x float> %RHS, <4 x i32> <i32 4, i32 0, i32 6, i32 6>
; CHECK-NEXT: ret <4 x float> %tmp4
  %tmp1 = extractelement <4 x float> %LHS, i32 0
  %tmp2 = insertelement <4 x float> %RHS, float %tmp1, i32 1
  %tmp3 = extractelement <4 x float> %RHS, i32 2
  %tmp4 = insertelement <4 x float> %tmp2, float %tmp3, i32 3
  ret <4 x float> %tmp4
}
 
define <4 x float> @test15b(<4 x float> %LHS, <4 x float> %RHS) {
; CHECK-LABEL: @test15b
; CHECK-NEXT: shufflevector <4 x float> %LHS, <4 x float> %RHS, <4 x i32> <i32 4, i32 3, i32 6, i32 6>
; CHECK-NEXT: ret <4 x float> %tmp5
  %tmp0 = extractelement <4 x float> %LHS, i32 3
  %tmp1 = insertelement <4 x float> %RHS, float %tmp0, i32 0
  %tmp2 = extractelement <4 x float> %tmp1, i32 0
  %tmp3 = insertelement <4 x float> %RHS, float %tmp2, i32 1
  %tmp4 = extractelement <4 x float> %RHS, i32 2
  %tmp5 = insertelement <4 x float> %tmp3, float %tmp4, i32 3
  ret <4 x float> %tmp5
}

define <1 x i32> @test16a(i32 %ele) {
; CHECK-LABEL: @test16a(
; CHECK-NEXT: ret <1 x i32> <i32 2>
  %tmp0 = insertelement <2 x i32> <i32 1, i32 undef>, i32 %ele, i32 1
  %tmp1 = shl <2 x i32> %tmp0, <i32 1, i32 1>
  %tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <1 x i32> <i32 0>
  ret <1 x i32> %tmp2
}

define <4 x i8> @test16b(i8 %ele) {
; CHECK-LABEL: @test16b(
; CHECK-NEXT: ret <4 x i8> <i8 2, i8 2, i8 2, i8 2>
  %tmp0 = insertelement <8 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 undef, i8 1>, i8 %ele, i32 6
  %tmp1 = shl <8 x i8> %tmp0, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  ret <4 x i8> %tmp2
}

; If composition of two shuffles is identity, shuffles can be removed.
define <4 x i32> @shuffle_17ident(<4 x i32> %v) nounwind uwtable {
; CHECK-LABEL: @shuffle_17ident(
; CHECK-NOT: shufflevector
  %shuffle = shufflevector <4 x i32> %v, <4 x i32> zeroinitializer,
                           <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  %shuffle2 = shufflevector <4 x i32> %shuffle, <4 x i32> zeroinitializer,
                            <4 x i32> <i32 3, i32 0, i32 1, i32 2>
  ret <4 x i32> %shuffle2
}

; swizzle can be put after operation
define <4 x i32> @shuffle_17and(<4 x i32> %v1, <4 x i32> %v2) nounwind uwtable {
; CHECK-LABEL: @shuffle_17and(
; CHECK-NOT: shufflevector
; CHECK: and <4 x i32> %v1, %v2
; CHECK: shufflevector
  %t1 = shufflevector <4 x i32> %v1, <4 x i32> zeroinitializer,
                      <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  %t2 = shufflevector <4 x i32> %v2, <4 x i32> zeroinitializer,
                      <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  %r = and <4 x i32> %t1, %t2
  ret <4 x i32> %r
}

define <4 x i32> @shuffle_17add(<4 x i32> %v1, <4 x i32> %v2) nounwind uwtable {
; CHECK-LABEL: @shuffle_17add(
; CHECK-NOT: shufflevector
; CHECK: add <4 x i32> %v1, %v2
; CHECK: shufflevector
  %t1 = shufflevector <4 x i32> %v1, <4 x i32> zeroinitializer,
                      <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  %t2 = shufflevector <4 x i32> %v2, <4 x i32> zeroinitializer,
                      <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  %r = add <4 x i32> %t1, %t2
  ret <4 x i32> %r
}

define <4 x i32> @shuffle_17addnsw(<4 x i32> %v1, <4 x i32> %v2) nounwind uwtable {
; CHECK-LABEL: @shuffle_17addnsw(
; CHECK-NOT: shufflevector
; CHECK: add nsw <4 x i32> %v1, %v2
; CHECK: shufflevector
  %t1 = shufflevector <4 x i32> %v1, <4 x i32> zeroinitializer,
                      <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  %t2 = shufflevector <4 x i32> %v2, <4 x i32> zeroinitializer,
                      <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  %r = add nsw <4 x i32> %t1, %t2
  ret <4 x i32> %r
}

define <4 x i32> @shuffle_17addnuw(<4 x i32> %v1, <4 x i32> %v2) nounwind uwtable {
; CHECK-LABEL: @shuffle_17addnuw(
; CHECK-NOT: shufflevector
; CHECK: add nuw <4 x i32> %v1, %v2
; CHECK: shufflevector
  %t1 = shufflevector <4 x i32> %v1, <4 x i32> zeroinitializer,
                      <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  %t2 = shufflevector <4 x i32> %v2, <4 x i32> zeroinitializer,
                      <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  %r = add nuw <4 x i32> %t1, %t2
  ret <4 x i32> %r
}

define <4 x float> @shuffle_17fsub_fast(<4 x float> %v1, <4 x float> %v2) nounwind uwtable {
; CHECK-LABEL: @shuffle_17fsub_fast(
; CHECK-NEXT: [[VAR1:%[a-zA-Z0-9.]+]] = fsub fast <4 x float> %v1, %v2
; CHECK-NEXT: shufflevector <4 x float> [[VAR1]], <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0>
; CHECK-NEXT: ret <4 x float>
  %t1 = shufflevector <4 x float> %v1, <4 x float> zeroinitializer,
                      <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  %t2 = shufflevector <4 x float> %v2, <4 x float> zeroinitializer,
                      <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  %r = fsub fast <4 x float> %t1, %t2
  ret <4 x float> %r
}

define <4 x i32> @shuffle_17addconst(<4 x i32> %v1, <4 x i32> %v2) {
; CHECK-LABEL: @shuffle_17addconst(
; CHECK-NOT: shufflevector
; CHECK: [[VAR1:%[a-zA-Z0-9.]+]] = add <4 x i32> %v1, <i32 4, i32 1, i32 2, i32 3>
; CHECK: [[VAR2:%[a-zA-Z0-9.]+]] = shufflevector <4 x i32> [[VAR1]], <4 x i32> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0>
; CHECK: ret <4 x i32> [[VAR2]]
  %t1 = shufflevector <4 x i32> %v1, <4 x i32> zeroinitializer,
                      <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  %r = add <4 x i32> %t1, <i32 1, i32 2, i32 3, i32 4>
  ret <4 x i32> %r
}

define <4 x i32> @shuffle_17add2(<4 x i32> %v) {
; CHECK-LABEL: @shuffle_17add2(
; CHECK-NOT: shufflevector
; CHECK: [[VAR:%[a-zA-Z0-9.]+]] = shl <4 x i32> %v, <i32 1, i32 1, i32 1, i32 1>
; CHECK: ret <4 x i32> [[VAR]]
  %t1 = shufflevector <4 x i32> %v, <4 x i32> zeroinitializer,
                      <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %t2 = add <4 x i32> %t1, %t1
  %r = shufflevector <4 x i32> %t2, <4 x i32> zeroinitializer,
                     <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i32> %r
}

define <4 x i32> @shuffle_17mulsplat(<4 x i32> %v) {
; CHECK-LABEL: @shuffle_17mulsplat(
; CHECK-NOT: shufflevector
; CHECK: [[VAR1:%[a-zA-Z0-9.]+]] = mul <4 x i32> %v, %v
; CHECK: [[VAR2:%[a-zA-Z0-9.]+]] = shufflevector <4 x i32> [[VAR1]], <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK: ret <4 x i32> [[VAR2]]
  %s1 = shufflevector <4 x i32> %v,
                      <4 x i32> zeroinitializer,
                      <4 x i32> zeroinitializer
  %m1 = mul <4 x i32> %s1, %s1
  %s2 = shufflevector <4 x i32> %m1,
                      <4 x i32> zeroinitializer,
                      <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %s2
}

; Do not reorder shuffle and binop if LHS of shuffles are of different size
define <2 x i32> @pr19717(<4 x i32> %in0, <2 x i32> %in1) {
; CHECK-LABEL: @pr19717(
; CHECK: shufflevector
; CHECK: shufflevector
; CHECK: mul
  %shuffle = shufflevector <4 x i32> %in0, <4 x i32> %in0, <2 x i32> zeroinitializer
  %shuffle4 = shufflevector <2 x i32> %in1, <2 x i32> %in1, <2 x i32> zeroinitializer
  %mul = mul <2 x i32> %shuffle, %shuffle4
  ret <2 x i32> %mul
}

define <4 x i16> @pr19717a(<8 x i16> %in0, <8 x i16> %in1) {
; CHECK-LABEL: @pr19717a(
; CHECK: [[VAR1:%[a-zA-Z0-9.]+]] = mul <8 x i16> %in0, %in1
; CHECK: [[VAR2:%[a-zA-Z0-9.]+]] = shufflevector <8 x i16> [[VAR1]], <8 x i16> undef, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
; CHECK: ret <4 x i16> [[VAR2]]
  %shuffle = shufflevector <8 x i16> %in0, <8 x i16> %in0, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %shuffle1 = shufflevector <8 x i16> %in1, <8 x i16> %in1, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mul = mul <4 x i16> %shuffle, %shuffle1
  ret <4 x i16> %mul
}

define <8 x i8> @pr19730(<16 x i8> %in0) {
; CHECK-LABEL: @pr19730(
; CHECK: shufflevector
  %shuffle = shufflevector <16 x i8> %in0, <16 x i8> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %shuffle1 = shufflevector <8 x i8> %shuffle, <8 x i8> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i8> %shuffle1
}

define i32 @pr19737(<4 x i32> %in0) {
; CHECK-LABEL: @pr19737(
; CHECK: [[VAR:%[a-zA-Z0-9.]+]] = extractelement <4 x i32> %in0, i32 0
; CHECK: ret i32 [[VAR]]
  %shuffle.i = shufflevector <4 x i32> zeroinitializer, <4 x i32> %in0, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  %neg.i = xor <4 x i32> %shuffle.i, <i32 -1, i32 -1, i32 -1, i32 -1>
  %and.i = and <4 x i32> %in0, %neg.i
  %rv = extractelement <4 x i32> %and.i, i32 0
  ret i32 %rv
}

; In PR20059 ( http://llvm.org/pr20059 ), shufflevector operations are reordered/removed
; for an srem operation. This is not a valid optimization because it may cause a trap
; on div-by-zero.

define <4 x i32> @pr20059(<4 x i32> %p1, <4 x i32> %p2) {
; CHECK-LABEL: @pr20059(
; CHECK-NEXT: %splat1 = shufflevector <4 x i32> %p1, <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK-NEXT: %splat2 = shufflevector <4 x i32> %p2, <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK-NEXT: %retval = srem <4 x i32> %splat1, %splat2
  %splat1 = shufflevector <4 x i32> %p1, <4 x i32> undef, <4 x i32> zeroinitializer
  %splat2 = shufflevector <4 x i32> %p2, <4 x i32> undef, <4 x i32> zeroinitializer
  %retval = srem <4 x i32> %splat1, %splat2
  ret <4 x i32> %retval
}

define <4 x i32> @pr20114(<4 x i32> %__mask) {
; CHECK-LABEL: @pr20114
; CHECK: shufflevector
; CHECK: and
  %mask01.i = shufflevector <4 x i32> %__mask, <4 x i32> undef, <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  %masked_new.i.i.i = and <4 x i32> bitcast (<2 x i64> <i64 ptrtoint (<4 x i32> (<4 x i32>)* @pr20114 to i64), i64 ptrtoint (<4 x i32> (<4 x i32>)* @pr20114 to i64)> to <4 x i32>), %mask01.i
  ret <4 x i32> %masked_new.i.i.i
}

define <2 x i32*> @pr23113(<4 x i32*> %A) {
; CHECK-LABEL: @pr23113
; CHECK: %[[V:.*]] = shufflevector <4 x i32*> %A, <4 x i32*> undef, <2 x i32> <i32 0, i32 1>
; CHECK-NEXT: ret <2 x i32*> %[[V]]
  %1 = shufflevector <4 x i32*> %A, <4 x i32*> undef, <2 x i32> <i32 0, i32 1>
  ret <2 x i32*> %1
}
