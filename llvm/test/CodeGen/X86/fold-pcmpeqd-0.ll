; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck --check-prefix=X86-64 %s
; DISABLED: llc < %s -mtriple=i386-apple-darwin -mcpu=yonah -regalloc=linearscan | FileCheck --check-prefix=I386 %s

; i386 test has been disabled when scheduler 2-addr hack is disabled.

; This testcase shouldn't need to spill the -1 value,
; so it should just use pcmpeqd to materialize an all-ones vector.
; For i386, cp load of -1 are folded.

; With -regalloc=greedy, the live range is split before spilling, so the first
; pcmpeq doesn't get folded as a constant pool load.

; I386-NOT: pcmpeqd
; I386: orps LCPI0_2, %xmm
; I386-NOT: pcmpeqd
; I386: orps LCPI0_2, %xmm

; X86-64: pcmpeqd
; X86-64-NOT: pcmpeqd

	%struct.__ImageExecInfo = type <{ <4 x i32>, <4 x float>, <2 x i64>, i8*, i8*, i8*, i32, i32, i32, i32, i32 }>
	%struct._cl_image_format_t = type <{ i32, i32, i32 }>
	%struct._image2d_t = type <{ i8*, %struct._cl_image_format_t, i32, i32, i32, i32, i32, i32 }>

define void @program_1(%struct._image2d_t* %dest, %struct._image2d_t* %t0, <4 x float> %p0, <4 x float> %p1, <4 x float> %p4, <4 x float> %p5, <4 x float> %p6) nounwind {
entry:
	%tmp3.i = load i32* null		; <i32> [#uses=1]
	%cmp = icmp sgt i32 %tmp3.i, 200		; <i1> [#uses=1]
	br i1 %cmp, label %forcond, label %ifthen

ifthen:		; preds = %entry
	ret void

forcond:		; preds = %entry
	%tmp3.i536 = load i32* null		; <i32> [#uses=1]
	%cmp12 = icmp slt i32 0, %tmp3.i536		; <i1> [#uses=1]
	br i1 %cmp12, label %forbody, label %afterfor

forbody:		; preds = %forcond
	%bitcast204.i313 = bitcast <4 x i32> zeroinitializer to <4 x float>		; <<4 x float>> [#uses=1]
	%mul233 = fmul <4 x float> %bitcast204.i313, zeroinitializer		; <<4 x float>> [#uses=1]
	%mul257 = fmul <4 x float> %mul233, zeroinitializer		; <<4 x float>> [#uses=1]
	%mul275 = fmul <4 x float> %mul257, zeroinitializer		; <<4 x float>> [#uses=1]
	%tmp51 = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %mul275, <4 x float> zeroinitializer) nounwind		; <<4 x float>> [#uses=1]
	%bitcast198.i182 = bitcast <4 x float> zeroinitializer to <4 x i32>		; <<4 x i32>> [#uses=0]
	%bitcast204.i185 = bitcast <4 x i32> zeroinitializer to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp69 = call <4 x i32> @llvm.x86.sse2.cvttps2dq(<4 x float> zeroinitializer) nounwind		; <<4 x i32>> [#uses=1]
	%tmp70 = call <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32> %tmp69) nounwind		; <<4 x float>> [#uses=1]
	%sub140.i78 = fsub <4 x float> zeroinitializer, %tmp70		; <<4 x float>> [#uses=2]
	%mul166.i86 = fmul <4 x float> zeroinitializer, %sub140.i78		; <<4 x float>> [#uses=1]
	%add167.i87 = fadd <4 x float> %mul166.i86, < float 0x3FE62ACB60000000, float 0x3FE62ACB60000000, float 0x3FE62ACB60000000, float 0x3FE62ACB60000000 >		; <<4 x float>> [#uses=1]
	%mul171.i88 = fmul <4 x float> %add167.i87, %sub140.i78		; <<4 x float>> [#uses=1]
	%add172.i89 = fadd <4 x float> %mul171.i88, < float 0x3FF0000A40000000, float 0x3FF0000A40000000, float 0x3FF0000A40000000, float 0x3FF0000A40000000 >		; <<4 x float>> [#uses=1]
	%bitcast176.i90 = bitcast <4 x float> %add172.i89 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%andnps178.i92 = and <4 x i32> %bitcast176.i90, zeroinitializer		; <<4 x i32>> [#uses=1]
	%bitcast179.i93 = bitcast <4 x i32> %andnps178.i92 to <4 x float>		; <<4 x float>> [#uses=1]
	%mul186.i96 = fmul <4 x float> %bitcast179.i93, zeroinitializer		; <<4 x float>> [#uses=1]
	%bitcast190.i98 = bitcast <4 x float> %mul186.i96 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%andnps192.i100 = and <4 x i32> %bitcast190.i98, zeroinitializer		; <<4 x i32>> [#uses=1]
	%xorps.i102 = xor <4 x i32> zeroinitializer, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%orps203.i103 = or <4 x i32> %andnps192.i100, %xorps.i102		; <<4 x i32>> [#uses=1]
	%bitcast204.i104 = bitcast <4 x i32> %orps203.i103 to <4 x float>		; <<4 x float>> [#uses=1]
	%cmple.i = call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> zeroinitializer, <4 x float> %tmp51, i8 2) nounwind		; <<4 x float>> [#uses=1]
	%tmp80 = call <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32> zeroinitializer) nounwind		; <<4 x float>> [#uses=1]
	%sub140.i = fsub <4 x float> zeroinitializer, %tmp80		; <<4 x float>> [#uses=1]
	%bitcast148.i = bitcast <4 x float> zeroinitializer to <4 x i32>		; <<4 x i32>> [#uses=1]
	%andnps150.i = and <4 x i32> %bitcast148.i, < i32 -2139095041, i32 -2139095041, i32 -2139095041, i32 -2139095041 >		; <<4 x i32>> [#uses=0]
	%mul171.i = fmul <4 x float> zeroinitializer, %sub140.i		; <<4 x float>> [#uses=1]
	%add172.i = fadd <4 x float> %mul171.i, < float 0x3FF0000A40000000, float 0x3FF0000A40000000, float 0x3FF0000A40000000, float 0x3FF0000A40000000 >		; <<4 x float>> [#uses=1]
	%bitcast176.i = bitcast <4 x float> %add172.i to <4 x i32>		; <<4 x i32>> [#uses=1]
	%andnps178.i = and <4 x i32> %bitcast176.i, zeroinitializer		; <<4 x i32>> [#uses=1]
	%bitcast179.i = bitcast <4 x i32> %andnps178.i to <4 x float>		; <<4 x float>> [#uses=1]
	%mul186.i = fmul <4 x float> %bitcast179.i, zeroinitializer		; <<4 x float>> [#uses=1]
	%bitcast189.i = bitcast <4 x float> zeroinitializer to <4 x i32>		; <<4 x i32>> [#uses=0]
	%bitcast190.i = bitcast <4 x float> %mul186.i to <4 x i32>		; <<4 x i32>> [#uses=1]
	%andnps192.i = and <4 x i32> %bitcast190.i, zeroinitializer		; <<4 x i32>> [#uses=1]
	%bitcast198.i = bitcast <4 x float> %cmple.i to <4 x i32>		; <<4 x i32>> [#uses=1]
	%xorps.i = xor <4 x i32> %bitcast198.i, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%orps203.i = or <4 x i32> %andnps192.i, %xorps.i		; <<4 x i32>> [#uses=1]
	%bitcast204.i = bitcast <4 x i32> %orps203.i to <4 x float>		; <<4 x float>> [#uses=1]
	%mul307 = fmul <4 x float> %bitcast204.i185, zeroinitializer		; <<4 x float>> [#uses=1]
	%mul310 = fmul <4 x float> %bitcast204.i104, zeroinitializer		; <<4 x float>> [#uses=2]
	%mul313 = fmul <4 x float> %bitcast204.i, zeroinitializer		; <<4 x float>> [#uses=1]
	%tmp82 = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %mul307, <4 x float> zeroinitializer) nounwind		; <<4 x float>> [#uses=1]
	%bitcast11.i15 = bitcast <4 x float> %tmp82 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%andnps.i17 = and <4 x i32> %bitcast11.i15, zeroinitializer		; <<4 x i32>> [#uses=1]
	%orps.i18 = or <4 x i32> %andnps.i17, zeroinitializer		; <<4 x i32>> [#uses=1]
	%bitcast17.i19 = bitcast <4 x i32> %orps.i18 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp83 = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %mul310, <4 x float> zeroinitializer) nounwind		; <<4 x float>> [#uses=1]
	%bitcast.i3 = bitcast <4 x float> %mul310 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%bitcast6.i4 = bitcast <4 x float> zeroinitializer to <4 x i32>		; <<4 x i32>> [#uses=2]
	%andps.i5 = and <4 x i32> %bitcast.i3, %bitcast6.i4		; <<4 x i32>> [#uses=1]
	%bitcast11.i6 = bitcast <4 x float> %tmp83 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%not.i7 = xor <4 x i32> %bitcast6.i4, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%andnps.i8 = and <4 x i32> %bitcast11.i6, %not.i7		; <<4 x i32>> [#uses=1]
	%orps.i9 = or <4 x i32> %andnps.i8, %andps.i5		; <<4 x i32>> [#uses=1]
	%bitcast17.i10 = bitcast <4 x i32> %orps.i9 to <4 x float>		; <<4 x float>> [#uses=1]
	%bitcast.i = bitcast <4 x float> %mul313 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%andps.i = and <4 x i32> %bitcast.i, zeroinitializer		; <<4 x i32>> [#uses=1]
	%orps.i = or <4 x i32> zeroinitializer, %andps.i		; <<4 x i32>> [#uses=1]
	%bitcast17.i = bitcast <4 x i32> %orps.i to <4 x float>		; <<4 x float>> [#uses=1]
	call void null(<4 x float> %bitcast17.i19, <4 x float> %bitcast17.i10, <4 x float> %bitcast17.i, <4 x float> zeroinitializer, %struct.__ImageExecInfo* null, <4 x i32> zeroinitializer) nounwind
	unreachable

afterfor:		; preds = %forcond
	ret void
}

declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8) nounwind readnone

declare <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32>) nounwind readnone

declare <4 x i32> @llvm.x86.sse2.cvttps2dq(<4 x float>) nounwind readnone

declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone

declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone
