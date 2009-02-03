; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx

;; A basic sanity check to make sure that MMX arithmetic actually compiles.

define void @foo(<8 x i8>* %A, <8 x i8>* %B) {
entry:
	%tmp1 = load <8 x i8>* %A		; <<8 x i8>> [#uses=1]
	%tmp3 = load <8 x i8>* %B		; <<8 x i8>> [#uses=1]
	%tmp4 = add <8 x i8> %tmp1, %tmp3		; <<8 x i8>> [#uses=2]
	store <8 x i8> %tmp4, <8 x i8>* %A
	%tmp7 = load <8 x i8>* %B		; <<8 x i8>> [#uses=1]
	%tmp12 = tail call <8 x i8> @llvm.x86.mmx.padds.b( <8 x i8> %tmp4, <8 x i8> %tmp7 )		; <<8 x i8>> [#uses=2]
	store <8 x i8> %tmp12, <8 x i8>* %A
	%tmp16 = load <8 x i8>* %B		; <<8 x i8>> [#uses=1]
	%tmp21 = tail call <8 x i8> @llvm.x86.mmx.paddus.b( <8 x i8> %tmp12, <8 x i8> %tmp16 )		; <<8 x i8>> [#uses=2]
	store <8 x i8> %tmp21, <8 x i8>* %A
	%tmp27 = load <8 x i8>* %B		; <<8 x i8>> [#uses=1]
	%tmp28 = sub <8 x i8> %tmp21, %tmp27		; <<8 x i8>> [#uses=2]
	store <8 x i8> %tmp28, <8 x i8>* %A
	%tmp31 = load <8 x i8>* %B		; <<8 x i8>> [#uses=1]
	%tmp36 = tail call <8 x i8> @llvm.x86.mmx.psubs.b( <8 x i8> %tmp28, <8 x i8> %tmp31 )		; <<8 x i8>> [#uses=2]
	store <8 x i8> %tmp36, <8 x i8>* %A
	%tmp40 = load <8 x i8>* %B		; <<8 x i8>> [#uses=1]
	%tmp45 = tail call <8 x i8> @llvm.x86.mmx.psubus.b( <8 x i8> %tmp36, <8 x i8> %tmp40 )		; <<8 x i8>> [#uses=2]
	store <8 x i8> %tmp45, <8 x i8>* %A
	%tmp51 = load <8 x i8>* %B		; <<8 x i8>> [#uses=1]
	%tmp52 = mul <8 x i8> %tmp45, %tmp51		; <<8 x i8>> [#uses=2]
	store <8 x i8> %tmp52, <8 x i8>* %A
	%tmp57 = load <8 x i8>* %B		; <<8 x i8>> [#uses=1]
	%tmp58 = and <8 x i8> %tmp52, %tmp57		; <<8 x i8>> [#uses=2]
	store <8 x i8> %tmp58, <8 x i8>* %A
	%tmp63 = load <8 x i8>* %B		; <<8 x i8>> [#uses=1]
	%tmp64 = or <8 x i8> %tmp58, %tmp63		; <<8 x i8>> [#uses=2]
	store <8 x i8> %tmp64, <8 x i8>* %A
	%tmp69 = load <8 x i8>* %B		; <<8 x i8>> [#uses=1]
	%tmp70 = xor <8 x i8> %tmp64, %tmp69		; <<8 x i8>> [#uses=1]
	store <8 x i8> %tmp70, <8 x i8>* %A
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

define void @baz(<2 x i32>* %A, <2 x i32>* %B) {
entry:
	%tmp1 = load <2 x i32>* %A		; <<2 x i32>> [#uses=1]
	%tmp3 = load <2 x i32>* %B		; <<2 x i32>> [#uses=1]
	%tmp4 = add <2 x i32> %tmp1, %tmp3		; <<2 x i32>> [#uses=2]
	store <2 x i32> %tmp4, <2 x i32>* %A
	%tmp9 = load <2 x i32>* %B		; <<2 x i32>> [#uses=1]
	%tmp10 = sub <2 x i32> %tmp4, %tmp9		; <<2 x i32>> [#uses=2]
	store <2 x i32> %tmp10, <2 x i32>* %A
	%tmp15 = load <2 x i32>* %B		; <<2 x i32>> [#uses=1]
	%tmp16 = mul <2 x i32> %tmp10, %tmp15		; <<2 x i32>> [#uses=2]
	store <2 x i32> %tmp16, <2 x i32>* %A
	%tmp21 = load <2 x i32>* %B		; <<2 x i32>> [#uses=1]
	%tmp22 = and <2 x i32> %tmp16, %tmp21		; <<2 x i32>> [#uses=2]
	store <2 x i32> %tmp22, <2 x i32>* %A
	%tmp27 = load <2 x i32>* %B		; <<2 x i32>> [#uses=1]
	%tmp28 = or <2 x i32> %tmp22, %tmp27		; <<2 x i32>> [#uses=2]
	store <2 x i32> %tmp28, <2 x i32>* %A
	%tmp33 = load <2 x i32>* %B		; <<2 x i32>> [#uses=1]
	%tmp34 = xor <2 x i32> %tmp28, %tmp33		; <<2 x i32>> [#uses=1]
	store <2 x i32> %tmp34, <2 x i32>* %A
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

define void @bar(<4 x i16>* %A, <4 x i16>* %B) {
entry:
	%tmp1 = load <4 x i16>* %A		; <<4 x i16>> [#uses=1]
	%tmp3 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp4 = add <4 x i16> %tmp1, %tmp3		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp4, <4 x i16>* %A
	%tmp7 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp12 = tail call <4 x i16> @llvm.x86.mmx.padds.w( <4 x i16> %tmp4, <4 x i16> %tmp7 )		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp12, <4 x i16>* %A
	%tmp16 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp21 = tail call <4 x i16> @llvm.x86.mmx.paddus.w( <4 x i16> %tmp12, <4 x i16> %tmp16 )		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp21, <4 x i16>* %A
	%tmp27 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp28 = sub <4 x i16> %tmp21, %tmp27		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp28, <4 x i16>* %A
	%tmp31 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp36 = tail call <4 x i16> @llvm.x86.mmx.psubs.w( <4 x i16> %tmp28, <4 x i16> %tmp31 )		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp36, <4 x i16>* %A
	%tmp40 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp45 = tail call <4 x i16> @llvm.x86.mmx.psubus.w( <4 x i16> %tmp36, <4 x i16> %tmp40 )		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp45, <4 x i16>* %A
	%tmp51 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp52 = mul <4 x i16> %tmp45, %tmp51		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp52, <4 x i16>* %A
	%tmp55 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp60 = tail call <4 x i16> @llvm.x86.mmx.pmulh.w( <4 x i16> %tmp52, <4 x i16> %tmp55 )		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp60, <4 x i16>* %A
	%tmp64 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp69 = tail call <2 x i32> @llvm.x86.mmx.pmadd.wd( <4 x i16> %tmp60, <4 x i16> %tmp64 )		; <<2 x i32>> [#uses=1]
	%tmp70 = bitcast <2 x i32> %tmp69 to <4 x i16>		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp70, <4 x i16>* %A
	%tmp75 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp76 = and <4 x i16> %tmp70, %tmp75		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp76, <4 x i16>* %A
	%tmp81 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp82 = or <4 x i16> %tmp76, %tmp81		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp82, <4 x i16>* %A
	%tmp87 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp88 = xor <4 x i16> %tmp82, %tmp87		; <<4 x i16>> [#uses=1]
	store <4 x i16> %tmp88, <4 x i16>* %A
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

declare <8 x i8> @llvm.x86.mmx.padds.b(<8 x i8>, <8 x i8>)

declare <8 x i8> @llvm.x86.mmx.paddus.b(<8 x i8>, <8 x i8>)

declare <8 x i8> @llvm.x86.mmx.psubs.b(<8 x i8>, <8 x i8>)

declare <8 x i8> @llvm.x86.mmx.psubus.b(<8 x i8>, <8 x i8>)

declare <4 x i16> @llvm.x86.mmx.padds.w(<4 x i16>, <4 x i16>)

declare <4 x i16> @llvm.x86.mmx.paddus.w(<4 x i16>, <4 x i16>)

declare <4 x i16> @llvm.x86.mmx.psubs.w(<4 x i16>, <4 x i16>)

declare <4 x i16> @llvm.x86.mmx.psubus.w(<4 x i16>, <4 x i16>)

declare <4 x i16> @llvm.x86.mmx.pmulh.w(<4 x i16>, <4 x i16>)

declare <2 x i32> @llvm.x86.mmx.pmadd.wd(<4 x i16>, <4 x i16>)

declare void @llvm.x86.mmx.emms()
