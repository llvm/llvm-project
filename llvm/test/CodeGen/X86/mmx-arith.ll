; RUN: llc < %s -march=x86 -mattr=+mmx

;; A basic sanity check to make sure that MMX arithmetic actually compiles.
;; First is a straight translation of the original with bitcasts as needed.

define void @foo(x86_mmx* %A, x86_mmx* %B) {
entry:
	%tmp1 = load x86_mmx* %A		; <x86_mmx> [#uses=1]
	%tmp3 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp1a = bitcast x86_mmx %tmp1 to <8 x i8>
        %tmp3a = bitcast x86_mmx %tmp3 to <8 x i8>
	%tmp4 = add <8 x i8> %tmp1a, %tmp3a		; <<8 x i8>> [#uses=2]
        %tmp4a = bitcast <8 x i8> %tmp4 to x86_mmx
	store x86_mmx %tmp4a, x86_mmx* %A
	%tmp7 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp12 = tail call x86_mmx @llvm.x86.mmx.padds.b( x86_mmx %tmp4a, x86_mmx %tmp7 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp12, x86_mmx* %A
	%tmp16 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp21 = tail call x86_mmx @llvm.x86.mmx.paddus.b( x86_mmx %tmp12, x86_mmx %tmp16 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp21, x86_mmx* %A
	%tmp27 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp21a = bitcast x86_mmx %tmp21 to <8 x i8>
        %tmp27a = bitcast x86_mmx %tmp27 to <8 x i8>
	%tmp28 = sub <8 x i8> %tmp21a, %tmp27a		; <<8 x i8>> [#uses=2]
        %tmp28a = bitcast <8 x i8> %tmp28 to x86_mmx
	store x86_mmx %tmp28a, x86_mmx* %A
	%tmp31 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp36 = tail call x86_mmx @llvm.x86.mmx.psubs.b( x86_mmx %tmp28a, x86_mmx %tmp31 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp36, x86_mmx* %A
	%tmp40 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp45 = tail call x86_mmx @llvm.x86.mmx.psubus.b( x86_mmx %tmp36, x86_mmx %tmp40 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp45, x86_mmx* %A
	%tmp51 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp45a = bitcast x86_mmx %tmp45 to <8 x i8>
        %tmp51a = bitcast x86_mmx %tmp51 to <8 x i8>
	%tmp52 = mul <8 x i8> %tmp45a, %tmp51a		; <<8 x i8>> [#uses=2]
        %tmp52a = bitcast <8 x i8> %tmp52 to x86_mmx
	store x86_mmx %tmp52a, x86_mmx* %A
	%tmp57 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp57a = bitcast x86_mmx %tmp57 to <8 x i8>
	%tmp58 = and <8 x i8> %tmp52, %tmp57a		; <<8 x i8>> [#uses=2]
        %tmp58a = bitcast <8 x i8> %tmp58 to x86_mmx
	store x86_mmx %tmp58a, x86_mmx* %A
	%tmp63 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp63a = bitcast x86_mmx %tmp63 to <8 x i8>
	%tmp64 = or <8 x i8> %tmp58, %tmp63a		; <<8 x i8>> [#uses=2]
        %tmp64a = bitcast <8 x i8> %tmp64 to x86_mmx
	store x86_mmx %tmp64a, x86_mmx* %A
	%tmp69 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp69a = bitcast x86_mmx %tmp69 to <8 x i8>
        %tmp64b = bitcast x86_mmx %tmp64a to <8 x i8>
	%tmp70 = xor <8 x i8> %tmp64b, %tmp69a		; <<8 x i8>> [#uses=1]
        %tmp70a = bitcast <8 x i8> %tmp70 to x86_mmx
	store x86_mmx %tmp70a, x86_mmx* %A
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

define void @baz(x86_mmx* %A, x86_mmx* %B) {
entry:
	%tmp1 = load x86_mmx* %A		; <x86_mmx> [#uses=1]
	%tmp3 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp1a = bitcast x86_mmx %tmp1 to <2 x i32>
        %tmp3a = bitcast x86_mmx %tmp3 to <2 x i32>
	%tmp4 = add <2 x i32> %tmp1a, %tmp3a		; <<2 x i32>> [#uses=2]
        %tmp4a = bitcast <2 x i32> %tmp4 to x86_mmx
	store x86_mmx %tmp4a, x86_mmx* %A
	%tmp9 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp9a = bitcast x86_mmx %tmp9 to <2 x i32>
	%tmp10 = sub <2 x i32> %tmp4, %tmp9a		; <<2 x i32>> [#uses=2]
        %tmp10a = bitcast <2 x i32> %tmp4 to x86_mmx
	store x86_mmx %tmp10a, x86_mmx* %A
	%tmp15 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp10b = bitcast x86_mmx %tmp10a to <2 x i32>
        %tmp15a = bitcast x86_mmx %tmp15 to <2 x i32>
	%tmp16 = mul <2 x i32> %tmp10b, %tmp15a		; <<2 x i32>> [#uses=2]
        %tmp16a = bitcast <2 x i32> %tmp16 to x86_mmx
	store x86_mmx %tmp16a, x86_mmx* %A
	%tmp21 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp16b = bitcast x86_mmx %tmp16a to <2 x i32>
        %tmp21a = bitcast x86_mmx %tmp21 to <2 x i32>
	%tmp22 = and <2 x i32> %tmp16b, %tmp21a		; <<2 x i32>> [#uses=2]
        %tmp22a = bitcast <2 x i32> %tmp22 to x86_mmx
	store x86_mmx %tmp22a, x86_mmx* %A
	%tmp27 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp22b = bitcast x86_mmx %tmp22a to <2 x i32>
        %tmp27a = bitcast x86_mmx %tmp27 to <2 x i32>
	%tmp28 = or <2 x i32> %tmp22b, %tmp27a		; <<2 x i32>> [#uses=2]
        %tmp28a = bitcast <2 x i32> %tmp28 to x86_mmx
	store x86_mmx %tmp28a, x86_mmx* %A
	%tmp33 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp28b = bitcast x86_mmx %tmp28a to <2 x i32>
        %tmp33a = bitcast x86_mmx %tmp33 to <2 x i32>
	%tmp34 = xor <2 x i32> %tmp28b, %tmp33a		; <<2 x i32>> [#uses=1]
        %tmp34a = bitcast <2 x i32> %tmp34 to x86_mmx
	store x86_mmx %tmp34a, x86_mmx* %A
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

define void @bar(x86_mmx* %A, x86_mmx* %B) {
entry:
	%tmp1 = load x86_mmx* %A		; <x86_mmx> [#uses=1]
	%tmp3 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp1a = bitcast x86_mmx %tmp1 to <4 x i16>
        %tmp3a = bitcast x86_mmx %tmp3 to <4 x i16>
	%tmp4 = add <4 x i16> %tmp1a, %tmp3a		; <<4 x i16>> [#uses=2]
        %tmp4a = bitcast <4 x i16> %tmp4 to x86_mmx
	store x86_mmx %tmp4a, x86_mmx* %A
	%tmp7 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp12 = tail call x86_mmx @llvm.x86.mmx.padds.w( x86_mmx %tmp4a, x86_mmx %tmp7 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp12, x86_mmx* %A
	%tmp16 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp21 = tail call x86_mmx @llvm.x86.mmx.paddus.w( x86_mmx %tmp12, x86_mmx %tmp16 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp21, x86_mmx* %A
	%tmp27 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp21a = bitcast x86_mmx %tmp21 to <4 x i16>
        %tmp27a = bitcast x86_mmx %tmp27 to <4 x i16>
	%tmp28 = sub <4 x i16> %tmp21a, %tmp27a		; <<4 x i16>> [#uses=2]
        %tmp28a = bitcast <4 x i16> %tmp28 to x86_mmx
	store x86_mmx %tmp28a, x86_mmx* %A
	%tmp31 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp36 = tail call x86_mmx @llvm.x86.mmx.psubs.w( x86_mmx %tmp28a, x86_mmx %tmp31 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp36, x86_mmx* %A
	%tmp40 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp45 = tail call x86_mmx @llvm.x86.mmx.psubus.w( x86_mmx %tmp36, x86_mmx %tmp40 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp45, x86_mmx* %A
	%tmp51 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp45a = bitcast x86_mmx %tmp45 to <4 x i16>
        %tmp51a = bitcast x86_mmx %tmp51 to <4 x i16>
	%tmp52 = mul <4 x i16> %tmp45a, %tmp51a		; <<4 x i16>> [#uses=2]
        %tmp52a = bitcast <4 x i16> %tmp52 to x86_mmx
	store x86_mmx %tmp52a, x86_mmx* %A
	%tmp55 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp60 = tail call x86_mmx @llvm.x86.mmx.pmulh.w( x86_mmx %tmp52a, x86_mmx %tmp55 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp60, x86_mmx* %A
	%tmp64 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp69 = tail call x86_mmx @llvm.x86.mmx.pmadd.wd( x86_mmx %tmp60, x86_mmx %tmp64 )		; <x86_mmx> [#uses=1]
	%tmp70 = bitcast x86_mmx %tmp69 to x86_mmx		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp70, x86_mmx* %A
	%tmp75 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp70a = bitcast x86_mmx %tmp70 to <4 x i16>
        %tmp75a = bitcast x86_mmx %tmp75 to <4 x i16>
	%tmp76 = and <4 x i16> %tmp70a, %tmp75a		; <<4 x i16>> [#uses=2]
        %tmp76a = bitcast <4 x i16> %tmp76 to x86_mmx
	store x86_mmx %tmp76a, x86_mmx* %A
	%tmp81 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp76b = bitcast x86_mmx %tmp76a to <4 x i16>
        %tmp81a = bitcast x86_mmx %tmp81 to <4 x i16>
	%tmp82 = or <4 x i16> %tmp76b, %tmp81a		; <<4 x i16>> [#uses=2]
        %tmp82a = bitcast <4 x i16> %tmp82 to x86_mmx
	store x86_mmx %tmp82a, x86_mmx* %A
	%tmp87 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp82b = bitcast x86_mmx %tmp82a to <4 x i16>
        %tmp87a = bitcast x86_mmx %tmp87 to <4 x i16>
	%tmp88 = xor <4 x i16> %tmp82b, %tmp87a		; <<4 x i16>> [#uses=1]
        %tmp88a = bitcast <4 x i16> %tmp88 to x86_mmx
	store x86_mmx %tmp88a, x86_mmx* %A
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

;; The following is modified to use MMX intrinsics everywhere they work.

define void @fooa(x86_mmx* %A, x86_mmx* %B) {
entry:
	%tmp1 = load x86_mmx* %A		; <x86_mmx> [#uses=1]
	%tmp3 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp4 = tail call x86_mmx @llvm.x86.mmx.padd.b( x86_mmx %tmp1, x86_mmx %tmp3 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp4, x86_mmx* %A
	%tmp7 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp12 = tail call x86_mmx @llvm.x86.mmx.padds.b( x86_mmx %tmp4, x86_mmx %tmp7 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp12, x86_mmx* %A
	%tmp16 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp21 = tail call x86_mmx @llvm.x86.mmx.paddus.b( x86_mmx %tmp12, x86_mmx %tmp16 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp21, x86_mmx* %A
	%tmp27 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp28 = tail call x86_mmx @llvm.x86.mmx.psub.b( x86_mmx %tmp21, x86_mmx %tmp27 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp28, x86_mmx* %A
	%tmp31 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp36 = tail call x86_mmx @llvm.x86.mmx.psubs.b( x86_mmx %tmp28, x86_mmx %tmp31 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp36, x86_mmx* %A
	%tmp40 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp45 = tail call x86_mmx @llvm.x86.mmx.psubus.b( x86_mmx %tmp36, x86_mmx %tmp40 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp45, x86_mmx* %A
	%tmp51 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp51a = bitcast x86_mmx %tmp51 to i64
        %tmp51aa = bitcast i64 %tmp51a to <8 x i8>
        %tmp51b = bitcast x86_mmx %tmp45 to <8 x i8>
	%tmp52 = mul <8 x i8> %tmp51b, %tmp51aa		; <x86_mmx> [#uses=2]
        %tmp52a = bitcast <8 x i8> %tmp52 to i64
        %tmp52aa = bitcast i64 %tmp52a to x86_mmx
	store x86_mmx %tmp52aa, x86_mmx* %A
	%tmp57 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp58 = tail call x86_mmx @llvm.x86.mmx.pand( x86_mmx %tmp51, x86_mmx %tmp57 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp58, x86_mmx* %A
	%tmp63 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp64 = tail call x86_mmx @llvm.x86.mmx.por( x86_mmx %tmp58, x86_mmx %tmp63 )		; <x86_mmx> [#uses=2]	
	store x86_mmx %tmp64, x86_mmx* %A
	%tmp69 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp70 = tail call x86_mmx @llvm.x86.mmx.pxor( x86_mmx %tmp64, x86_mmx %tmp69 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp70, x86_mmx* %A
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

define void @baza(x86_mmx* %A, x86_mmx* %B) {
entry:
	%tmp1 = load x86_mmx* %A		; <x86_mmx> [#uses=1]
	%tmp3 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp4 = tail call x86_mmx @llvm.x86.mmx.padd.d( x86_mmx %tmp1, x86_mmx %tmp3 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp4, x86_mmx* %A
	%tmp9 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp10 = tail call x86_mmx @llvm.x86.mmx.psub.d( x86_mmx %tmp4, x86_mmx %tmp9 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp10, x86_mmx* %A
	%tmp15 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
        %tmp10a = bitcast x86_mmx %tmp10 to <2 x i32>
        %tmp15a = bitcast x86_mmx %tmp15 to <2 x i32>
	%tmp16 = mul <2 x i32> %tmp10a, %tmp15a		; <x86_mmx> [#uses=2]
        %tmp16a = bitcast <2 x i32> %tmp16 to x86_mmx
	store x86_mmx %tmp16a, x86_mmx* %A
	%tmp21 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp22 = tail call x86_mmx @llvm.x86.mmx.pand( x86_mmx %tmp16a, x86_mmx %tmp21 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp22, x86_mmx* %A
	%tmp27 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp28 = tail call x86_mmx @llvm.x86.mmx.por( x86_mmx %tmp22, x86_mmx %tmp27 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp28, x86_mmx* %A
	%tmp33 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp34 = tail call x86_mmx @llvm.x86.mmx.pxor( x86_mmx %tmp28, x86_mmx %tmp33 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp34, x86_mmx* %A
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

define void @bara(x86_mmx* %A, x86_mmx* %B) {
entry:
	%tmp1 = load x86_mmx* %A		; <x86_mmx> [#uses=1]
	%tmp3 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp4 = tail call x86_mmx @llvm.x86.mmx.padd.w( x86_mmx %tmp1, x86_mmx %tmp3 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp4, x86_mmx* %A
	%tmp7 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp12 = tail call x86_mmx @llvm.x86.mmx.padds.w( x86_mmx %tmp4, x86_mmx %tmp7 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp12, x86_mmx* %A
	%tmp16 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp21 = tail call x86_mmx @llvm.x86.mmx.paddus.w( x86_mmx %tmp12, x86_mmx %tmp16 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp21, x86_mmx* %A
	%tmp27 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp28 = tail call x86_mmx @llvm.x86.mmx.psub.w( x86_mmx %tmp21, x86_mmx %tmp27 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp28, x86_mmx* %A
	%tmp31 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp36 = tail call x86_mmx @llvm.x86.mmx.psubs.w( x86_mmx %tmp28, x86_mmx %tmp31 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp36, x86_mmx* %A
	%tmp40 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp45 = tail call x86_mmx @llvm.x86.mmx.psubus.w( x86_mmx %tmp36, x86_mmx %tmp40 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp45, x86_mmx* %A
	%tmp51 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp52 = tail call x86_mmx @llvm.x86.mmx.pmull.w( x86_mmx %tmp45, x86_mmx %tmp51 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp52, x86_mmx* %A
	%tmp55 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp60 = tail call x86_mmx @llvm.x86.mmx.pmulh.w( x86_mmx %tmp52, x86_mmx %tmp55 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp60, x86_mmx* %A
	%tmp64 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp69 = tail call x86_mmx @llvm.x86.mmx.pmadd.wd( x86_mmx %tmp60, x86_mmx %tmp64 )		; <x86_mmx> [#uses=1]
	%tmp70 = bitcast x86_mmx %tmp69 to x86_mmx		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp70, x86_mmx* %A
	%tmp75 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp76 = tail call x86_mmx @llvm.x86.mmx.pand( x86_mmx %tmp70, x86_mmx %tmp75 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp76, x86_mmx* %A
	%tmp81 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp82 = tail call x86_mmx @llvm.x86.mmx.por( x86_mmx %tmp76, x86_mmx %tmp81 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp82, x86_mmx* %A
	%tmp87 = load x86_mmx* %B		; <x86_mmx> [#uses=1]
	%tmp88 = tail call x86_mmx @llvm.x86.mmx.pxor( x86_mmx %tmp82, x86_mmx %tmp87 )		; <x86_mmx> [#uses=2]
	store x86_mmx %tmp88, x86_mmx* %A
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

declare x86_mmx @llvm.x86.mmx.paddus.b(x86_mmx, x86_mmx)

declare x86_mmx @llvm.x86.mmx.psubus.b(x86_mmx, x86_mmx)

declare x86_mmx @llvm.x86.mmx.paddus.w(x86_mmx, x86_mmx)

declare x86_mmx @llvm.x86.mmx.psubus.w(x86_mmx, x86_mmx)

declare x86_mmx @llvm.x86.mmx.pmulh.w(x86_mmx, x86_mmx)

declare x86_mmx @llvm.x86.mmx.pmadd.wd(x86_mmx, x86_mmx)

declare void @llvm.x86.mmx.emms()

declare x86_mmx @llvm.x86.mmx.padd.b(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.padd.w(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.padd.d(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.padds.b(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.padds.w(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.padds.d(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.psubs.b(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.psubs.w(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.psubs.d(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.psub.b(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.psub.w(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.psub.d(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.pmull.w(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.pand(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.por(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.pxor(x86_mmx, x86_mmx)

