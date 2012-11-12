; RUN: opt < %s -instcombine | \
; RUN:   llc -march=ppc32 -mcpu=g5 | not grep vperm
; RUN: llc < %s -march=ppc32 -mcpu=g5 > %t
; RUN: grep vsldoi  %t | count 2
; RUN: grep vmrgh   %t | count 7
; RUN: grep vmrgl   %t | count 6
; RUN: grep vpkuhum %t | count 1
; RUN: grep vpkuwum %t | count 1

define void @VSLDOI_xy(<8 x i16>* %A, <8 x i16>* %B) {
entry:
	%tmp = load <8 x i16>* %A		; <<8 x i16>> [#uses=1]
	%tmp2 = load <8 x i16>* %B		; <<8 x i16>> [#uses=1]
	%tmp.upgrd.1 = bitcast <8 x i16> %tmp to <16 x i8>		; <<16 x i8>> [#uses=11]
	%tmp2.upgrd.2 = bitcast <8 x i16> %tmp2 to <16 x i8>		; <<16 x i8>> [#uses=5]
	%tmp.upgrd.3 = extractelement <16 x i8> %tmp.upgrd.1, i32 5		; <i8> [#uses=1]
	%tmp3 = extractelement <16 x i8> %tmp.upgrd.1, i32 6		; <i8> [#uses=1]
	%tmp4 = extractelement <16 x i8> %tmp.upgrd.1, i32 7		; <i8> [#uses=1]
	%tmp5 = extractelement <16 x i8> %tmp.upgrd.1, i32 8		; <i8> [#uses=1]
	%tmp6 = extractelement <16 x i8> %tmp.upgrd.1, i32 9		; <i8> [#uses=1]
	%tmp7 = extractelement <16 x i8> %tmp.upgrd.1, i32 10		; <i8> [#uses=1]
	%tmp8 = extractelement <16 x i8> %tmp.upgrd.1, i32 11		; <i8> [#uses=1]
	%tmp9 = extractelement <16 x i8> %tmp.upgrd.1, i32 12		; <i8> [#uses=1]
	%tmp10 = extractelement <16 x i8> %tmp.upgrd.1, i32 13		; <i8> [#uses=1]
	%tmp11 = extractelement <16 x i8> %tmp.upgrd.1, i32 14		; <i8> [#uses=1]
	%tmp12 = extractelement <16 x i8> %tmp.upgrd.1, i32 15		; <i8> [#uses=1]
	%tmp13 = extractelement <16 x i8> %tmp2.upgrd.2, i32 0		; <i8> [#uses=1]
	%tmp14 = extractelement <16 x i8> %tmp2.upgrd.2, i32 1		; <i8> [#uses=1]
	%tmp15 = extractelement <16 x i8> %tmp2.upgrd.2, i32 2		; <i8> [#uses=1]
	%tmp16 = extractelement <16 x i8> %tmp2.upgrd.2, i32 3		; <i8> [#uses=1]
	%tmp17 = extractelement <16 x i8> %tmp2.upgrd.2, i32 4		; <i8> [#uses=1]
	%tmp18 = insertelement <16 x i8> undef, i8 %tmp.upgrd.3, i32 0		; <<16 x i8>> [#uses=1]
	%tmp19 = insertelement <16 x i8> %tmp18, i8 %tmp3, i32 1		; <<16 x i8>> [#uses=1]
	%tmp20 = insertelement <16 x i8> %tmp19, i8 %tmp4, i32 2		; <<16 x i8>> [#uses=1]
	%tmp21 = insertelement <16 x i8> %tmp20, i8 %tmp5, i32 3		; <<16 x i8>> [#uses=1]
	%tmp22 = insertelement <16 x i8> %tmp21, i8 %tmp6, i32 4		; <<16 x i8>> [#uses=1]
	%tmp23 = insertelement <16 x i8> %tmp22, i8 %tmp7, i32 5		; <<16 x i8>> [#uses=1]
	%tmp24 = insertelement <16 x i8> %tmp23, i8 %tmp8, i32 6		; <<16 x i8>> [#uses=1]
	%tmp25 = insertelement <16 x i8> %tmp24, i8 %tmp9, i32 7		; <<16 x i8>> [#uses=1]
	%tmp26 = insertelement <16 x i8> %tmp25, i8 %tmp10, i32 8		; <<16 x i8>> [#uses=1]
	%tmp27 = insertelement <16 x i8> %tmp26, i8 %tmp11, i32 9		; <<16 x i8>> [#uses=1]
	%tmp28 = insertelement <16 x i8> %tmp27, i8 %tmp12, i32 10		; <<16 x i8>> [#uses=1]
	%tmp29 = insertelement <16 x i8> %tmp28, i8 %tmp13, i32 11		; <<16 x i8>> [#uses=1]
	%tmp30 = insertelement <16 x i8> %tmp29, i8 %tmp14, i32 12		; <<16 x i8>> [#uses=1]
	%tmp31 = insertelement <16 x i8> %tmp30, i8 %tmp15, i32 13		; <<16 x i8>> [#uses=1]
	%tmp32 = insertelement <16 x i8> %tmp31, i8 %tmp16, i32 14		; <<16 x i8>> [#uses=1]
	%tmp33 = insertelement <16 x i8> %tmp32, i8 %tmp17, i32 15		; <<16 x i8>> [#uses=1]
	%tmp33.upgrd.4 = bitcast <16 x i8> %tmp33 to <8 x i16>		; <<8 x i16>> [#uses=1]
	store <8 x i16> %tmp33.upgrd.4, <8 x i16>* %A
	ret void
}

define void @VSLDOI_xx(<8 x i16>* %A, <8 x i16>* %B) {
	%tmp = load <8 x i16>* %A		; <<8 x i16>> [#uses=1]
	%tmp2 = load <8 x i16>* %A		; <<8 x i16>> [#uses=1]
	%tmp.upgrd.5 = bitcast <8 x i16> %tmp to <16 x i8>		; <<16 x i8>> [#uses=11]
	%tmp2.upgrd.6 = bitcast <8 x i16> %tmp2 to <16 x i8>		; <<16 x i8>> [#uses=5]
	%tmp.upgrd.7 = extractelement <16 x i8> %tmp.upgrd.5, i32 5		; <i8> [#uses=1]
	%tmp3 = extractelement <16 x i8> %tmp.upgrd.5, i32 6		; <i8> [#uses=1]
	%tmp4 = extractelement <16 x i8> %tmp.upgrd.5, i32 7		; <i8> [#uses=1]
	%tmp5 = extractelement <16 x i8> %tmp.upgrd.5, i32 8		; <i8> [#uses=1]
	%tmp6 = extractelement <16 x i8> %tmp.upgrd.5, i32 9		; <i8> [#uses=1]
	%tmp7 = extractelement <16 x i8> %tmp.upgrd.5, i32 10		; <i8> [#uses=1]
	%tmp8 = extractelement <16 x i8> %tmp.upgrd.5, i32 11		; <i8> [#uses=1]
	%tmp9 = extractelement <16 x i8> %tmp.upgrd.5, i32 12		; <i8> [#uses=1]
	%tmp10 = extractelement <16 x i8> %tmp.upgrd.5, i32 13		; <i8> [#uses=1]
	%tmp11 = extractelement <16 x i8> %tmp.upgrd.5, i32 14		; <i8> [#uses=1]
	%tmp12 = extractelement <16 x i8> %tmp.upgrd.5, i32 15		; <i8> [#uses=1]
	%tmp13 = extractelement <16 x i8> %tmp2.upgrd.6, i32 0		; <i8> [#uses=1]
	%tmp14 = extractelement <16 x i8> %tmp2.upgrd.6, i32 1		; <i8> [#uses=1]
	%tmp15 = extractelement <16 x i8> %tmp2.upgrd.6, i32 2		; <i8> [#uses=1]
	%tmp16 = extractelement <16 x i8> %tmp2.upgrd.6, i32 3		; <i8> [#uses=1]
	%tmp17 = extractelement <16 x i8> %tmp2.upgrd.6, i32 4		; <i8> [#uses=1]
	%tmp18 = insertelement <16 x i8> undef, i8 %tmp.upgrd.7, i32 0		; <<16 x i8>> [#uses=1]
	%tmp19 = insertelement <16 x i8> %tmp18, i8 %tmp3, i32 1		; <<16 x i8>> [#uses=1]
	%tmp20 = insertelement <16 x i8> %tmp19, i8 %tmp4, i32 2		; <<16 x i8>> [#uses=1]
	%tmp21 = insertelement <16 x i8> %tmp20, i8 %tmp5, i32 3		; <<16 x i8>> [#uses=1]
	%tmp22 = insertelement <16 x i8> %tmp21, i8 %tmp6, i32 4		; <<16 x i8>> [#uses=1]
	%tmp23 = insertelement <16 x i8> %tmp22, i8 %tmp7, i32 5		; <<16 x i8>> [#uses=1]
	%tmp24 = insertelement <16 x i8> %tmp23, i8 %tmp8, i32 6		; <<16 x i8>> [#uses=1]
	%tmp25 = insertelement <16 x i8> %tmp24, i8 %tmp9, i32 7		; <<16 x i8>> [#uses=1]
	%tmp26 = insertelement <16 x i8> %tmp25, i8 %tmp10, i32 8		; <<16 x i8>> [#uses=1]
	%tmp27 = insertelement <16 x i8> %tmp26, i8 %tmp11, i32 9		; <<16 x i8>> [#uses=1]
	%tmp28 = insertelement <16 x i8> %tmp27, i8 %tmp12, i32 10		; <<16 x i8>> [#uses=1]
	%tmp29 = insertelement <16 x i8> %tmp28, i8 %tmp13, i32 11		; <<16 x i8>> [#uses=1]
	%tmp30 = insertelement <16 x i8> %tmp29, i8 %tmp14, i32 12		; <<16 x i8>> [#uses=1]
	%tmp31 = insertelement <16 x i8> %tmp30, i8 %tmp15, i32 13		; <<16 x i8>> [#uses=1]
	%tmp32 = insertelement <16 x i8> %tmp31, i8 %tmp16, i32 14		; <<16 x i8>> [#uses=1]
	%tmp33 = insertelement <16 x i8> %tmp32, i8 %tmp17, i32 15		; <<16 x i8>> [#uses=1]
	%tmp33.upgrd.8 = bitcast <16 x i8> %tmp33 to <8 x i16>		; <<8 x i16>> [#uses=1]
	store <8 x i16> %tmp33.upgrd.8, <8 x i16>* %A
	ret void
}

define void @VPERM_promote(<8 x i16>* %A, <8 x i16>* %B) {
entry:
	%tmp = load <8 x i16>* %A		; <<8 x i16>> [#uses=1]
	%tmp.upgrd.9 = bitcast <8 x i16> %tmp to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp2 = load <8 x i16>* %B		; <<8 x i16>> [#uses=1]
	%tmp2.upgrd.10 = bitcast <8 x i16> %tmp2 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp3 = call <4 x i32> @llvm.ppc.altivec.vperm( <4 x i32> %tmp.upgrd.9, <4 x i32> %tmp2.upgrd.10, <16 x i8> < i8 14, i8 14, i8 14, i8 14, i8 14, i8 14, i8 14, i8 14, i8 14, i8 14, i8 14, i8 14, i8 14, i8 14, i8 14, i8 14 > )		; <<4 x i32>> [#uses=1]
	%tmp3.upgrd.11 = bitcast <4 x i32> %tmp3 to <8 x i16>		; <<8 x i16>> [#uses=1]
	store <8 x i16> %tmp3.upgrd.11, <8 x i16>* %A
	ret void
}

declare <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32>, <4 x i32>, <16 x i8>)

define void @tb_l(<16 x i8>* %A, <16 x i8>* %B) {
entry:
	%tmp = load <16 x i8>* %A		; <<16 x i8>> [#uses=8]
	%tmp2 = load <16 x i8>* %B		; <<16 x i8>> [#uses=8]
	%tmp.upgrd.12 = extractelement <16 x i8> %tmp, i32 8		; <i8> [#uses=1]
	%tmp3 = extractelement <16 x i8> %tmp2, i32 8		; <i8> [#uses=1]
	%tmp4 = extractelement <16 x i8> %tmp, i32 9		; <i8> [#uses=1]
	%tmp5 = extractelement <16 x i8> %tmp2, i32 9		; <i8> [#uses=1]
	%tmp6 = extractelement <16 x i8> %tmp, i32 10		; <i8> [#uses=1]
	%tmp7 = extractelement <16 x i8> %tmp2, i32 10		; <i8> [#uses=1]
	%tmp8 = extractelement <16 x i8> %tmp, i32 11		; <i8> [#uses=1]
	%tmp9 = extractelement <16 x i8> %tmp2, i32 11		; <i8> [#uses=1]
	%tmp10 = extractelement <16 x i8> %tmp, i32 12		; <i8> [#uses=1]
	%tmp11 = extractelement <16 x i8> %tmp2, i32 12		; <i8> [#uses=1]
	%tmp12 = extractelement <16 x i8> %tmp, i32 13		; <i8> [#uses=1]
	%tmp13 = extractelement <16 x i8> %tmp2, i32 13		; <i8> [#uses=1]
	%tmp14 = extractelement <16 x i8> %tmp, i32 14		; <i8> [#uses=1]
	%tmp15 = extractelement <16 x i8> %tmp2, i32 14		; <i8> [#uses=1]
	%tmp16 = extractelement <16 x i8> %tmp, i32 15		; <i8> [#uses=1]
	%tmp17 = extractelement <16 x i8> %tmp2, i32 15		; <i8> [#uses=1]
	%tmp18 = insertelement <16 x i8> undef, i8 %tmp.upgrd.12, i32 0		; <<16 x i8>> [#uses=1]
	%tmp19 = insertelement <16 x i8> %tmp18, i8 %tmp3, i32 1		; <<16 x i8>> [#uses=1]
	%tmp20 = insertelement <16 x i8> %tmp19, i8 %tmp4, i32 2		; <<16 x i8>> [#uses=1]
	%tmp21 = insertelement <16 x i8> %tmp20, i8 %tmp5, i32 3		; <<16 x i8>> [#uses=1]
	%tmp22 = insertelement <16 x i8> %tmp21, i8 %tmp6, i32 4		; <<16 x i8>> [#uses=1]
	%tmp23 = insertelement <16 x i8> %tmp22, i8 %tmp7, i32 5		; <<16 x i8>> [#uses=1]
	%tmp24 = insertelement <16 x i8> %tmp23, i8 %tmp8, i32 6		; <<16 x i8>> [#uses=1]
	%tmp25 = insertelement <16 x i8> %tmp24, i8 %tmp9, i32 7		; <<16 x i8>> [#uses=1]
	%tmp26 = insertelement <16 x i8> %tmp25, i8 %tmp10, i32 8		; <<16 x i8>> [#uses=1]
	%tmp27 = insertelement <16 x i8> %tmp26, i8 %tmp11, i32 9		; <<16 x i8>> [#uses=1]
	%tmp28 = insertelement <16 x i8> %tmp27, i8 %tmp12, i32 10		; <<16 x i8>> [#uses=1]
	%tmp29 = insertelement <16 x i8> %tmp28, i8 %tmp13, i32 11		; <<16 x i8>> [#uses=1]
	%tmp30 = insertelement <16 x i8> %tmp29, i8 %tmp14, i32 12		; <<16 x i8>> [#uses=1]
	%tmp31 = insertelement <16 x i8> %tmp30, i8 %tmp15, i32 13		; <<16 x i8>> [#uses=1]
	%tmp32 = insertelement <16 x i8> %tmp31, i8 %tmp16, i32 14		; <<16 x i8>> [#uses=1]
	%tmp33 = insertelement <16 x i8> %tmp32, i8 %tmp17, i32 15		; <<16 x i8>> [#uses=1]
	store <16 x i8> %tmp33, <16 x i8>* %A
	ret void
}

define void @th_l(<8 x i16>* %A, <8 x i16>* %B) {
entry:
	%tmp = load <8 x i16>* %A		; <<8 x i16>> [#uses=4]
	%tmp2 = load <8 x i16>* %B		; <<8 x i16>> [#uses=4]
	%tmp.upgrd.13 = extractelement <8 x i16> %tmp, i32 4		; <i16> [#uses=1]
	%tmp3 = extractelement <8 x i16> %tmp2, i32 4		; <i16> [#uses=1]
	%tmp4 = extractelement <8 x i16> %tmp, i32 5		; <i16> [#uses=1]
	%tmp5 = extractelement <8 x i16> %tmp2, i32 5		; <i16> [#uses=1]
	%tmp6 = extractelement <8 x i16> %tmp, i32 6		; <i16> [#uses=1]
	%tmp7 = extractelement <8 x i16> %tmp2, i32 6		; <i16> [#uses=1]
	%tmp8 = extractelement <8 x i16> %tmp, i32 7		; <i16> [#uses=1]
	%tmp9 = extractelement <8 x i16> %tmp2, i32 7		; <i16> [#uses=1]
	%tmp10 = insertelement <8 x i16> undef, i16 %tmp.upgrd.13, i32 0		; <<8 x i16>> [#uses=1]
	%tmp11 = insertelement <8 x i16> %tmp10, i16 %tmp3, i32 1		; <<8 x i16>> [#uses=1]
	%tmp12 = insertelement <8 x i16> %tmp11, i16 %tmp4, i32 2		; <<8 x i16>> [#uses=1]
	%tmp13 = insertelement <8 x i16> %tmp12, i16 %tmp5, i32 3		; <<8 x i16>> [#uses=1]
	%tmp14 = insertelement <8 x i16> %tmp13, i16 %tmp6, i32 4		; <<8 x i16>> [#uses=1]
	%tmp15 = insertelement <8 x i16> %tmp14, i16 %tmp7, i32 5		; <<8 x i16>> [#uses=1]
	%tmp16 = insertelement <8 x i16> %tmp15, i16 %tmp8, i32 6		; <<8 x i16>> [#uses=1]
	%tmp17 = insertelement <8 x i16> %tmp16, i16 %tmp9, i32 7		; <<8 x i16>> [#uses=1]
	store <8 x i16> %tmp17, <8 x i16>* %A
	ret void
}

define void @tw_l(<4 x i32>* %A, <4 x i32>* %B) {
entry:
	%tmp = load <4 x i32>* %A		; <<4 x i32>> [#uses=2]
	%tmp2 = load <4 x i32>* %B		; <<4 x i32>> [#uses=2]
	%tmp.upgrd.14 = extractelement <4 x i32> %tmp, i32 2		; <i32> [#uses=1]
	%tmp3 = extractelement <4 x i32> %tmp2, i32 2		; <i32> [#uses=1]
	%tmp4 = extractelement <4 x i32> %tmp, i32 3		; <i32> [#uses=1]
	%tmp5 = extractelement <4 x i32> %tmp2, i32 3		; <i32> [#uses=1]
	%tmp6 = insertelement <4 x i32> undef, i32 %tmp.upgrd.14, i32 0		; <<4 x i32>> [#uses=1]
	%tmp7 = insertelement <4 x i32> %tmp6, i32 %tmp3, i32 1		; <<4 x i32>> [#uses=1]
	%tmp8 = insertelement <4 x i32> %tmp7, i32 %tmp4, i32 2		; <<4 x i32>> [#uses=1]
	%tmp9 = insertelement <4 x i32> %tmp8, i32 %tmp5, i32 3		; <<4 x i32>> [#uses=1]
	store <4 x i32> %tmp9, <4 x i32>* %A
	ret void
}

define void @tb_h(<16 x i8>* %A, <16 x i8>* %B) {
entry:
	%tmp = load <16 x i8>* %A		; <<16 x i8>> [#uses=8]
	%tmp2 = load <16 x i8>* %B		; <<16 x i8>> [#uses=8]
	%tmp.upgrd.15 = extractelement <16 x i8> %tmp, i32 0		; <i8> [#uses=1]
	%tmp3 = extractelement <16 x i8> %tmp2, i32 0		; <i8> [#uses=1]
	%tmp4 = extractelement <16 x i8> %tmp, i32 1		; <i8> [#uses=1]
	%tmp5 = extractelement <16 x i8> %tmp2, i32 1		; <i8> [#uses=1]
	%tmp6 = extractelement <16 x i8> %tmp, i32 2		; <i8> [#uses=1]
	%tmp7 = extractelement <16 x i8> %tmp2, i32 2		; <i8> [#uses=1]
	%tmp8 = extractelement <16 x i8> %tmp, i32 3		; <i8> [#uses=1]
	%tmp9 = extractelement <16 x i8> %tmp2, i32 3		; <i8> [#uses=1]
	%tmp10 = extractelement <16 x i8> %tmp, i32 4		; <i8> [#uses=1]
	%tmp11 = extractelement <16 x i8> %tmp2, i32 4		; <i8> [#uses=1]
	%tmp12 = extractelement <16 x i8> %tmp, i32 5		; <i8> [#uses=1]
	%tmp13 = extractelement <16 x i8> %tmp2, i32 5		; <i8> [#uses=1]
	%tmp14 = extractelement <16 x i8> %tmp, i32 6		; <i8> [#uses=1]
	%tmp15 = extractelement <16 x i8> %tmp2, i32 6		; <i8> [#uses=1]
	%tmp16 = extractelement <16 x i8> %tmp, i32 7		; <i8> [#uses=1]
	%tmp17 = extractelement <16 x i8> %tmp2, i32 7		; <i8> [#uses=1]
	%tmp18 = insertelement <16 x i8> undef, i8 %tmp.upgrd.15, i32 0		; <<16 x i8>> [#uses=1]
	%tmp19 = insertelement <16 x i8> %tmp18, i8 %tmp3, i32 1		; <<16 x i8>> [#uses=1]
	%tmp20 = insertelement <16 x i8> %tmp19, i8 %tmp4, i32 2		; <<16 x i8>> [#uses=1]
	%tmp21 = insertelement <16 x i8> %tmp20, i8 %tmp5, i32 3		; <<16 x i8>> [#uses=1]
	%tmp22 = insertelement <16 x i8> %tmp21, i8 %tmp6, i32 4		; <<16 x i8>> [#uses=1]
	%tmp23 = insertelement <16 x i8> %tmp22, i8 %tmp7, i32 5		; <<16 x i8>> [#uses=1]
	%tmp24 = insertelement <16 x i8> %tmp23, i8 %tmp8, i32 6		; <<16 x i8>> [#uses=1]
	%tmp25 = insertelement <16 x i8> %tmp24, i8 %tmp9, i32 7		; <<16 x i8>> [#uses=1]
	%tmp26 = insertelement <16 x i8> %tmp25, i8 %tmp10, i32 8		; <<16 x i8>> [#uses=1]
	%tmp27 = insertelement <16 x i8> %tmp26, i8 %tmp11, i32 9		; <<16 x i8>> [#uses=1]
	%tmp28 = insertelement <16 x i8> %tmp27, i8 %tmp12, i32 10		; <<16 x i8>> [#uses=1]
	%tmp29 = insertelement <16 x i8> %tmp28, i8 %tmp13, i32 11		; <<16 x i8>> [#uses=1]
	%tmp30 = insertelement <16 x i8> %tmp29, i8 %tmp14, i32 12		; <<16 x i8>> [#uses=1]
	%tmp31 = insertelement <16 x i8> %tmp30, i8 %tmp15, i32 13		; <<16 x i8>> [#uses=1]
	%tmp32 = insertelement <16 x i8> %tmp31, i8 %tmp16, i32 14		; <<16 x i8>> [#uses=1]
	%tmp33 = insertelement <16 x i8> %tmp32, i8 %tmp17, i32 15		; <<16 x i8>> [#uses=1]
	store <16 x i8> %tmp33, <16 x i8>* %A
	ret void
}

define void @th_h(<8 x i16>* %A, <8 x i16>* %B) {
entry:
	%tmp = load <8 x i16>* %A		; <<8 x i16>> [#uses=4]
	%tmp2 = load <8 x i16>* %B		; <<8 x i16>> [#uses=4]
	%tmp.upgrd.16 = extractelement <8 x i16> %tmp, i32 0		; <i16> [#uses=1]
	%tmp3 = extractelement <8 x i16> %tmp2, i32 0		; <i16> [#uses=1]
	%tmp4 = extractelement <8 x i16> %tmp, i32 1		; <i16> [#uses=1]
	%tmp5 = extractelement <8 x i16> %tmp2, i32 1		; <i16> [#uses=1]
	%tmp6 = extractelement <8 x i16> %tmp, i32 2		; <i16> [#uses=1]
	%tmp7 = extractelement <8 x i16> %tmp2, i32 2		; <i16> [#uses=1]
	%tmp8 = extractelement <8 x i16> %tmp, i32 3		; <i16> [#uses=1]
	%tmp9 = extractelement <8 x i16> %tmp2, i32 3		; <i16> [#uses=1]
	%tmp10 = insertelement <8 x i16> undef, i16 %tmp.upgrd.16, i32 0		; <<8 x i16>> [#uses=1]
	%tmp11 = insertelement <8 x i16> %tmp10, i16 %tmp3, i32 1		; <<8 x i16>> [#uses=1]
	%tmp12 = insertelement <8 x i16> %tmp11, i16 %tmp4, i32 2		; <<8 x i16>> [#uses=1]
	%tmp13 = insertelement <8 x i16> %tmp12, i16 %tmp5, i32 3		; <<8 x i16>> [#uses=1]
	%tmp14 = insertelement <8 x i16> %tmp13, i16 %tmp6, i32 4		; <<8 x i16>> [#uses=1]
	%tmp15 = insertelement <8 x i16> %tmp14, i16 %tmp7, i32 5		; <<8 x i16>> [#uses=1]
	%tmp16 = insertelement <8 x i16> %tmp15, i16 %tmp8, i32 6		; <<8 x i16>> [#uses=1]
	%tmp17 = insertelement <8 x i16> %tmp16, i16 %tmp9, i32 7		; <<8 x i16>> [#uses=1]
	store <8 x i16> %tmp17, <8 x i16>* %A
	ret void
}

define void @tw_h(<4 x i32>* %A, <4 x i32>* %B) {
entry:
	%tmp = load <4 x i32>* %A		; <<4 x i32>> [#uses=2]
	%tmp2 = load <4 x i32>* %B		; <<4 x i32>> [#uses=2]
	%tmp.upgrd.17 = extractelement <4 x i32> %tmp2, i32 0		; <i32> [#uses=1]
	%tmp3 = extractelement <4 x i32> %tmp, i32 0		; <i32> [#uses=1]
	%tmp4 = extractelement <4 x i32> %tmp2, i32 1		; <i32> [#uses=1]
	%tmp5 = extractelement <4 x i32> %tmp, i32 1		; <i32> [#uses=1]
	%tmp6 = insertelement <4 x i32> undef, i32 %tmp.upgrd.17, i32 0		; <<4 x i32>> [#uses=1]
	%tmp7 = insertelement <4 x i32> %tmp6, i32 %tmp3, i32 1		; <<4 x i32>> [#uses=1]
	%tmp8 = insertelement <4 x i32> %tmp7, i32 %tmp4, i32 2		; <<4 x i32>> [#uses=1]
	%tmp9 = insertelement <4 x i32> %tmp8, i32 %tmp5, i32 3		; <<4 x i32>> [#uses=1]
	store <4 x i32> %tmp9, <4 x i32>* %A
	ret void
}

define void @tw_h_flop(<4 x i32>* %A, <4 x i32>* %B) {
	%tmp = load <4 x i32>* %A		; <<4 x i32>> [#uses=2]
	%tmp2 = load <4 x i32>* %B		; <<4 x i32>> [#uses=2]
	%tmp.upgrd.18 = extractelement <4 x i32> %tmp, i32 0		; <i32> [#uses=1]
	%tmp3 = extractelement <4 x i32> %tmp2, i32 0		; <i32> [#uses=1]
	%tmp4 = extractelement <4 x i32> %tmp, i32 1		; <i32> [#uses=1]
	%tmp5 = extractelement <4 x i32> %tmp2, i32 1		; <i32> [#uses=1]
	%tmp6 = insertelement <4 x i32> undef, i32 %tmp.upgrd.18, i32 0		; <<4 x i32>> [#uses=1]
	%tmp7 = insertelement <4 x i32> %tmp6, i32 %tmp3, i32 1		; <<4 x i32>> [#uses=1]
	%tmp8 = insertelement <4 x i32> %tmp7, i32 %tmp4, i32 2		; <<4 x i32>> [#uses=1]
	%tmp9 = insertelement <4 x i32> %tmp8, i32 %tmp5, i32 3		; <<4 x i32>> [#uses=1]
	store <4 x i32> %tmp9, <4 x i32>* %A
	ret void
}

define void @VMRG_UNARY_tb_l(<16 x i8>* %A, <16 x i8>* %B) {
entry:
	%tmp = load <16 x i8>* %A		; <<16 x i8>> [#uses=16]
	%tmp.upgrd.19 = extractelement <16 x i8> %tmp, i32 8		; <i8> [#uses=1]
	%tmp3 = extractelement <16 x i8> %tmp, i32 8		; <i8> [#uses=1]
	%tmp4 = extractelement <16 x i8> %tmp, i32 9		; <i8> [#uses=1]
	%tmp5 = extractelement <16 x i8> %tmp, i32 9		; <i8> [#uses=1]
	%tmp6 = extractelement <16 x i8> %tmp, i32 10		; <i8> [#uses=1]
	%tmp7 = extractelement <16 x i8> %tmp, i32 10		; <i8> [#uses=1]
	%tmp8 = extractelement <16 x i8> %tmp, i32 11		; <i8> [#uses=1]
	%tmp9 = extractelement <16 x i8> %tmp, i32 11		; <i8> [#uses=1]
	%tmp10 = extractelement <16 x i8> %tmp, i32 12		; <i8> [#uses=1]
	%tmp11 = extractelement <16 x i8> %tmp, i32 12		; <i8> [#uses=1]
	%tmp12 = extractelement <16 x i8> %tmp, i32 13		; <i8> [#uses=1]
	%tmp13 = extractelement <16 x i8> %tmp, i32 13		; <i8> [#uses=1]
	%tmp14 = extractelement <16 x i8> %tmp, i32 14		; <i8> [#uses=1]
	%tmp15 = extractelement <16 x i8> %tmp, i32 14		; <i8> [#uses=1]
	%tmp16 = extractelement <16 x i8> %tmp, i32 15		; <i8> [#uses=1]
	%tmp17 = extractelement <16 x i8> %tmp, i32 15		; <i8> [#uses=1]
	%tmp18 = insertelement <16 x i8> undef, i8 %tmp.upgrd.19, i32 0		; <<16 x i8>> [#uses=1]
	%tmp19 = insertelement <16 x i8> %tmp18, i8 %tmp3, i32 1		; <<16 x i8>> [#uses=1]
	%tmp20 = insertelement <16 x i8> %tmp19, i8 %tmp4, i32 2		; <<16 x i8>> [#uses=1]
	%tmp21 = insertelement <16 x i8> %tmp20, i8 %tmp5, i32 3		; <<16 x i8>> [#uses=1]
	%tmp22 = insertelement <16 x i8> %tmp21, i8 %tmp6, i32 4		; <<16 x i8>> [#uses=1]
	%tmp23 = insertelement <16 x i8> %tmp22, i8 %tmp7, i32 5		; <<16 x i8>> [#uses=1]
	%tmp24 = insertelement <16 x i8> %tmp23, i8 %tmp8, i32 6		; <<16 x i8>> [#uses=1]
	%tmp25 = insertelement <16 x i8> %tmp24, i8 %tmp9, i32 7		; <<16 x i8>> [#uses=1]
	%tmp26 = insertelement <16 x i8> %tmp25, i8 %tmp10, i32 8		; <<16 x i8>> [#uses=1]
	%tmp27 = insertelement <16 x i8> %tmp26, i8 %tmp11, i32 9		; <<16 x i8>> [#uses=1]
	%tmp28 = insertelement <16 x i8> %tmp27, i8 %tmp12, i32 10		; <<16 x i8>> [#uses=1]
	%tmp29 = insertelement <16 x i8> %tmp28, i8 %tmp13, i32 11		; <<16 x i8>> [#uses=1]
	%tmp30 = insertelement <16 x i8> %tmp29, i8 %tmp14, i32 12		; <<16 x i8>> [#uses=1]
	%tmp31 = insertelement <16 x i8> %tmp30, i8 %tmp15, i32 13		; <<16 x i8>> [#uses=1]
	%tmp32 = insertelement <16 x i8> %tmp31, i8 %tmp16, i32 14		; <<16 x i8>> [#uses=1]
	%tmp33 = insertelement <16 x i8> %tmp32, i8 %tmp17, i32 15		; <<16 x i8>> [#uses=1]
	store <16 x i8> %tmp33, <16 x i8>* %A
	ret void
}

define void @VMRG_UNARY_th_l(<8 x i16>* %A, <8 x i16>* %B) {
entry:
	%tmp = load <8 x i16>* %A		; <<8 x i16>> [#uses=8]
	%tmp.upgrd.20 = extractelement <8 x i16> %tmp, i32 4		; <i16> [#uses=1]
	%tmp3 = extractelement <8 x i16> %tmp, i32 4		; <i16> [#uses=1]
	%tmp4 = extractelement <8 x i16> %tmp, i32 5		; <i16> [#uses=1]
	%tmp5 = extractelement <8 x i16> %tmp, i32 5		; <i16> [#uses=1]
	%tmp6 = extractelement <8 x i16> %tmp, i32 6		; <i16> [#uses=1]
	%tmp7 = extractelement <8 x i16> %tmp, i32 6		; <i16> [#uses=1]
	%tmp8 = extractelement <8 x i16> %tmp, i32 7		; <i16> [#uses=1]
	%tmp9 = extractelement <8 x i16> %tmp, i32 7		; <i16> [#uses=1]
	%tmp10 = insertelement <8 x i16> undef, i16 %tmp.upgrd.20, i32 0		; <<8 x i16>> [#uses=1]
	%tmp11 = insertelement <8 x i16> %tmp10, i16 %tmp3, i32 1		; <<8 x i16>> [#uses=1]
	%tmp12 = insertelement <8 x i16> %tmp11, i16 %tmp4, i32 2		; <<8 x i16>> [#uses=1]
	%tmp13 = insertelement <8 x i16> %tmp12, i16 %tmp5, i32 3		; <<8 x i16>> [#uses=1]
	%tmp14 = insertelement <8 x i16> %tmp13, i16 %tmp6, i32 4		; <<8 x i16>> [#uses=1]
	%tmp15 = insertelement <8 x i16> %tmp14, i16 %tmp7, i32 5		; <<8 x i16>> [#uses=1]
	%tmp16 = insertelement <8 x i16> %tmp15, i16 %tmp8, i32 6		; <<8 x i16>> [#uses=1]
	%tmp17 = insertelement <8 x i16> %tmp16, i16 %tmp9, i32 7		; <<8 x i16>> [#uses=1]
	store <8 x i16> %tmp17, <8 x i16>* %A
	ret void
}

define void @VMRG_UNARY_tw_l(<4 x i32>* %A, <4 x i32>* %B) {
entry:
	%tmp = load <4 x i32>* %A		; <<4 x i32>> [#uses=4]
	%tmp.upgrd.21 = extractelement <4 x i32> %tmp, i32 2		; <i32> [#uses=1]
	%tmp3 = extractelement <4 x i32> %tmp, i32 2		; <i32> [#uses=1]
	%tmp4 = extractelement <4 x i32> %tmp, i32 3		; <i32> [#uses=1]
	%tmp5 = extractelement <4 x i32> %tmp, i32 3		; <i32> [#uses=1]
	%tmp6 = insertelement <4 x i32> undef, i32 %tmp.upgrd.21, i32 0		; <<4 x i32>> [#uses=1]
	%tmp7 = insertelement <4 x i32> %tmp6, i32 %tmp3, i32 1		; <<4 x i32>> [#uses=1]
	%tmp8 = insertelement <4 x i32> %tmp7, i32 %tmp4, i32 2		; <<4 x i32>> [#uses=1]
	%tmp9 = insertelement <4 x i32> %tmp8, i32 %tmp5, i32 3		; <<4 x i32>> [#uses=1]
	store <4 x i32> %tmp9, <4 x i32>* %A
	ret void
}

define void @VMRG_UNARY_tb_h(<16 x i8>* %A, <16 x i8>* %B) {
entry:
	%tmp = load <16 x i8>* %A		; <<16 x i8>> [#uses=16]
	%tmp.upgrd.22 = extractelement <16 x i8> %tmp, i32 0		; <i8> [#uses=1]
	%tmp3 = extractelement <16 x i8> %tmp, i32 0		; <i8> [#uses=1]
	%tmp4 = extractelement <16 x i8> %tmp, i32 1		; <i8> [#uses=1]
	%tmp5 = extractelement <16 x i8> %tmp, i32 1		; <i8> [#uses=1]
	%tmp6 = extractelement <16 x i8> %tmp, i32 2		; <i8> [#uses=1]
	%tmp7 = extractelement <16 x i8> %tmp, i32 2		; <i8> [#uses=1]
	%tmp8 = extractelement <16 x i8> %tmp, i32 3		; <i8> [#uses=1]
	%tmp9 = extractelement <16 x i8> %tmp, i32 3		; <i8> [#uses=1]
	%tmp10 = extractelement <16 x i8> %tmp, i32 4		; <i8> [#uses=1]
	%tmp11 = extractelement <16 x i8> %tmp, i32 4		; <i8> [#uses=1]
	%tmp12 = extractelement <16 x i8> %tmp, i32 5		; <i8> [#uses=1]
	%tmp13 = extractelement <16 x i8> %tmp, i32 5		; <i8> [#uses=1]
	%tmp14 = extractelement <16 x i8> %tmp, i32 6		; <i8> [#uses=1]
	%tmp15 = extractelement <16 x i8> %tmp, i32 6		; <i8> [#uses=1]
	%tmp16 = extractelement <16 x i8> %tmp, i32 7		; <i8> [#uses=1]
	%tmp17 = extractelement <16 x i8> %tmp, i32 7		; <i8> [#uses=1]
	%tmp18 = insertelement <16 x i8> undef, i8 %tmp.upgrd.22, i32 0		; <<16 x i8>> [#uses=1]
	%tmp19 = insertelement <16 x i8> %tmp18, i8 %tmp3, i32 1		; <<16 x i8>> [#uses=1]
	%tmp20 = insertelement <16 x i8> %tmp19, i8 %tmp4, i32 2		; <<16 x i8>> [#uses=1]
	%tmp21 = insertelement <16 x i8> %tmp20, i8 %tmp5, i32 3		; <<16 x i8>> [#uses=1]
	%tmp22 = insertelement <16 x i8> %tmp21, i8 %tmp6, i32 4		; <<16 x i8>> [#uses=1]
	%tmp23 = insertelement <16 x i8> %tmp22, i8 %tmp7, i32 5		; <<16 x i8>> [#uses=1]
	%tmp24 = insertelement <16 x i8> %tmp23, i8 %tmp8, i32 6		; <<16 x i8>> [#uses=1]
	%tmp25 = insertelement <16 x i8> %tmp24, i8 %tmp9, i32 7		; <<16 x i8>> [#uses=1]
	%tmp26 = insertelement <16 x i8> %tmp25, i8 %tmp10, i32 8		; <<16 x i8>> [#uses=1]
	%tmp27 = insertelement <16 x i8> %tmp26, i8 %tmp11, i32 9		; <<16 x i8>> [#uses=1]
	%tmp28 = insertelement <16 x i8> %tmp27, i8 %tmp12, i32 10		; <<16 x i8>> [#uses=1]
	%tmp29 = insertelement <16 x i8> %tmp28, i8 %tmp13, i32 11		; <<16 x i8>> [#uses=1]
	%tmp30 = insertelement <16 x i8> %tmp29, i8 %tmp14, i32 12		; <<16 x i8>> [#uses=1]
	%tmp31 = insertelement <16 x i8> %tmp30, i8 %tmp15, i32 13		; <<16 x i8>> [#uses=1]
	%tmp32 = insertelement <16 x i8> %tmp31, i8 %tmp16, i32 14		; <<16 x i8>> [#uses=1]
	%tmp33 = insertelement <16 x i8> %tmp32, i8 %tmp17, i32 15		; <<16 x i8>> [#uses=1]
	store <16 x i8> %tmp33, <16 x i8>* %A
	ret void
}

define void @VMRG_UNARY_th_h(<8 x i16>* %A, <8 x i16>* %B) {
entry:
	%tmp = load <8 x i16>* %A		; <<8 x i16>> [#uses=8]
	%tmp.upgrd.23 = extractelement <8 x i16> %tmp, i32 0		; <i16> [#uses=1]
	%tmp3 = extractelement <8 x i16> %tmp, i32 0		; <i16> [#uses=1]
	%tmp4 = extractelement <8 x i16> %tmp, i32 1		; <i16> [#uses=1]
	%tmp5 = extractelement <8 x i16> %tmp, i32 1		; <i16> [#uses=1]
	%tmp6 = extractelement <8 x i16> %tmp, i32 2		; <i16> [#uses=1]
	%tmp7 = extractelement <8 x i16> %tmp, i32 2		; <i16> [#uses=1]
	%tmp8 = extractelement <8 x i16> %tmp, i32 3		; <i16> [#uses=1]
	%tmp9 = extractelement <8 x i16> %tmp, i32 3		; <i16> [#uses=1]
	%tmp10 = insertelement <8 x i16> undef, i16 %tmp.upgrd.23, i32 0		; <<8 x i16>> [#uses=1]
	%tmp11 = insertelement <8 x i16> %tmp10, i16 %tmp3, i32 1		; <<8 x i16>> [#uses=1]
	%tmp12 = insertelement <8 x i16> %tmp11, i16 %tmp4, i32 2		; <<8 x i16>> [#uses=1]
	%tmp13 = insertelement <8 x i16> %tmp12, i16 %tmp5, i32 3		; <<8 x i16>> [#uses=1]
	%tmp14 = insertelement <8 x i16> %tmp13, i16 %tmp6, i32 4		; <<8 x i16>> [#uses=1]
	%tmp15 = insertelement <8 x i16> %tmp14, i16 %tmp7, i32 5		; <<8 x i16>> [#uses=1]
	%tmp16 = insertelement <8 x i16> %tmp15, i16 %tmp8, i32 6		; <<8 x i16>> [#uses=1]
	%tmp17 = insertelement <8 x i16> %tmp16, i16 %tmp9, i32 7		; <<8 x i16>> [#uses=1]
	store <8 x i16> %tmp17, <8 x i16>* %A
	ret void
}

define void @VMRG_UNARY_tw_h(<4 x i32>* %A, <4 x i32>* %B) {
entry:
	%tmp = load <4 x i32>* %A		; <<4 x i32>> [#uses=4]
	%tmp.upgrd.24 = extractelement <4 x i32> %tmp, i32 0		; <i32> [#uses=1]
	%tmp3 = extractelement <4 x i32> %tmp, i32 0		; <i32> [#uses=1]
	%tmp4 = extractelement <4 x i32> %tmp, i32 1		; <i32> [#uses=1]
	%tmp5 = extractelement <4 x i32> %tmp, i32 1		; <i32> [#uses=1]
	%tmp6 = insertelement <4 x i32> undef, i32 %tmp.upgrd.24, i32 0		; <<4 x i32>> [#uses=1]
	%tmp7 = insertelement <4 x i32> %tmp6, i32 %tmp3, i32 1		; <<4 x i32>> [#uses=1]
	%tmp8 = insertelement <4 x i32> %tmp7, i32 %tmp4, i32 2		; <<4 x i32>> [#uses=1]
	%tmp9 = insertelement <4 x i32> %tmp8, i32 %tmp5, i32 3		; <<4 x i32>> [#uses=1]
	store <4 x i32> %tmp9, <4 x i32>* %A
	ret void
}

define void @VPCKUHUM_unary(<8 x i16>* %A, <8 x i16>* %B) {
entry:
	%tmp = load <8 x i16>* %A		; <<8 x i16>> [#uses=2]
	%tmp.upgrd.25 = bitcast <8 x i16> %tmp to <16 x i8>		; <<16 x i8>> [#uses=8]
	%tmp3 = bitcast <8 x i16> %tmp to <16 x i8>		; <<16 x i8>> [#uses=8]
	%tmp.upgrd.26 = extractelement <16 x i8> %tmp.upgrd.25, i32 1		; <i8> [#uses=1]
	%tmp4 = extractelement <16 x i8> %tmp.upgrd.25, i32 3		; <i8> [#uses=1]
	%tmp5 = extractelement <16 x i8> %tmp.upgrd.25, i32 5		; <i8> [#uses=1]
	%tmp6 = extractelement <16 x i8> %tmp.upgrd.25, i32 7		; <i8> [#uses=1]
	%tmp7 = extractelement <16 x i8> %tmp.upgrd.25, i32 9		; <i8> [#uses=1]
	%tmp8 = extractelement <16 x i8> %tmp.upgrd.25, i32 11		; <i8> [#uses=1]
	%tmp9 = extractelement <16 x i8> %tmp.upgrd.25, i32 13		; <i8> [#uses=1]
	%tmp10 = extractelement <16 x i8> %tmp.upgrd.25, i32 15		; <i8> [#uses=1]
	%tmp11 = extractelement <16 x i8> %tmp3, i32 1		; <i8> [#uses=1]
	%tmp12 = extractelement <16 x i8> %tmp3, i32 3		; <i8> [#uses=1]
	%tmp13 = extractelement <16 x i8> %tmp3, i32 5		; <i8> [#uses=1]
	%tmp14 = extractelement <16 x i8> %tmp3, i32 7		; <i8> [#uses=1]
	%tmp15 = extractelement <16 x i8> %tmp3, i32 9		; <i8> [#uses=1]
	%tmp16 = extractelement <16 x i8> %tmp3, i32 11		; <i8> [#uses=1]
	%tmp17 = extractelement <16 x i8> %tmp3, i32 13		; <i8> [#uses=1]
	%tmp18 = extractelement <16 x i8> %tmp3, i32 15		; <i8> [#uses=1]
	%tmp19 = insertelement <16 x i8> undef, i8 %tmp.upgrd.26, i32 0		; <<16 x i8>> [#uses=1]
	%tmp20 = insertelement <16 x i8> %tmp19, i8 %tmp4, i32 1		; <<16 x i8>> [#uses=1]
	%tmp21 = insertelement <16 x i8> %tmp20, i8 %tmp5, i32 2		; <<16 x i8>> [#uses=1]
	%tmp22 = insertelement <16 x i8> %tmp21, i8 %tmp6, i32 3		; <<16 x i8>> [#uses=1]
	%tmp23 = insertelement <16 x i8> %tmp22, i8 %tmp7, i32 4		; <<16 x i8>> [#uses=1]
	%tmp24 = insertelement <16 x i8> %tmp23, i8 %tmp8, i32 5		; <<16 x i8>> [#uses=1]
	%tmp25 = insertelement <16 x i8> %tmp24, i8 %tmp9, i32 6		; <<16 x i8>> [#uses=1]
	%tmp26 = insertelement <16 x i8> %tmp25, i8 %tmp10, i32 7		; <<16 x i8>> [#uses=1]
	%tmp27 = insertelement <16 x i8> %tmp26, i8 %tmp11, i32 8		; <<16 x i8>> [#uses=1]
	%tmp28 = insertelement <16 x i8> %tmp27, i8 %tmp12, i32 9		; <<16 x i8>> [#uses=1]
	%tmp29 = insertelement <16 x i8> %tmp28, i8 %tmp13, i32 10		; <<16 x i8>> [#uses=1]
	%tmp30 = insertelement <16 x i8> %tmp29, i8 %tmp14, i32 11		; <<16 x i8>> [#uses=1]
	%tmp31 = insertelement <16 x i8> %tmp30, i8 %tmp15, i32 12		; <<16 x i8>> [#uses=1]
	%tmp32 = insertelement <16 x i8> %tmp31, i8 %tmp16, i32 13		; <<16 x i8>> [#uses=1]
	%tmp33 = insertelement <16 x i8> %tmp32, i8 %tmp17, i32 14		; <<16 x i8>> [#uses=1]
	%tmp34 = insertelement <16 x i8> %tmp33, i8 %tmp18, i32 15		; <<16 x i8>> [#uses=1]
	%tmp34.upgrd.27 = bitcast <16 x i8> %tmp34 to <8 x i16>		; <<8 x i16>> [#uses=1]
	store <8 x i16> %tmp34.upgrd.27, <8 x i16>* %A
	ret void
}

define void @VPCKUWUM_unary(<4 x i32>* %A, <4 x i32>* %B) {
entry:
	%tmp = load <4 x i32>* %A		; <<4 x i32>> [#uses=2]
	%tmp.upgrd.28 = bitcast <4 x i32> %tmp to <8 x i16>		; <<8 x i16>> [#uses=4]
	%tmp3 = bitcast <4 x i32> %tmp to <8 x i16>		; <<8 x i16>> [#uses=4]
	%tmp.upgrd.29 = extractelement <8 x i16> %tmp.upgrd.28, i32 1		; <i16> [#uses=1]
	%tmp4 = extractelement <8 x i16> %tmp.upgrd.28, i32 3		; <i16> [#uses=1]
	%tmp5 = extractelement <8 x i16> %tmp.upgrd.28, i32 5		; <i16> [#uses=1]
	%tmp6 = extractelement <8 x i16> %tmp.upgrd.28, i32 7		; <i16> [#uses=1]
	%tmp7 = extractelement <8 x i16> %tmp3, i32 1		; <i16> [#uses=1]
	%tmp8 = extractelement <8 x i16> %tmp3, i32 3		; <i16> [#uses=1]
	%tmp9 = extractelement <8 x i16> %tmp3, i32 5		; <i16> [#uses=1]
	%tmp10 = extractelement <8 x i16> %tmp3, i32 7		; <i16> [#uses=1]
	%tmp11 = insertelement <8 x i16> undef, i16 %tmp.upgrd.29, i32 0		; <<8 x i16>> [#uses=1]
	%tmp12 = insertelement <8 x i16> %tmp11, i16 %tmp4, i32 1		; <<8 x i16>> [#uses=1]
	%tmp13 = insertelement <8 x i16> %tmp12, i16 %tmp5, i32 2		; <<8 x i16>> [#uses=1]
	%tmp14 = insertelement <8 x i16> %tmp13, i16 %tmp6, i32 3		; <<8 x i16>> [#uses=1]
	%tmp15 = insertelement <8 x i16> %tmp14, i16 %tmp7, i32 4		; <<8 x i16>> [#uses=1]
	%tmp16 = insertelement <8 x i16> %tmp15, i16 %tmp8, i32 5		; <<8 x i16>> [#uses=1]
	%tmp17 = insertelement <8 x i16> %tmp16, i16 %tmp9, i32 6		; <<8 x i16>> [#uses=1]
	%tmp18 = insertelement <8 x i16> %tmp17, i16 %tmp10, i32 7		; <<8 x i16>> [#uses=1]
	%tmp18.upgrd.30 = bitcast <8 x i16> %tmp18 to <4 x i32>		; <<4 x i32>> [#uses=1]
	store <4 x i32> %tmp18.upgrd.30, <4 x i32>* %A
	ret void
}
