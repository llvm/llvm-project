; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin -mcpu=yonah  | not grep pcmpeqd
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin | grep pcmpeqd | count 1

; This testcase should need to spill the -1 value on x86-32,
; so it shouldn't use pcmpeqd to materialize an all-ones vector; it
; should use a constant-pool load instead.

	%struct.__ImageExecInfo = type <{ <4 x i32>, <4 x float>, <2 x i64>, i8*, i8*, i8*, i32, i32, i32, i32, i32 }>
	%struct._cl_image_format_t = type <{ i32, i32, i32 }>
	%struct._image2d_t = type <{ i8*, %struct._cl_image_format_t, i32, i32, i32, i32, i32, i32 }>

define void @program_1(%struct._image2d_t* %dest, %struct._image2d_t* %t0, <4 x float> %p0, <4 x float> %p1, <4 x float> %p4, <4 x float> %p5, <4 x float> %p6) nounwind {
entry:
	%tmp3.i = load i32* null		; <i32> [#uses=1]
	%cmp = icmp slt i32 0, %tmp3.i		; <i1> [#uses=1]
	br i1 %cmp, label %forcond, label %ifthen

ifthen:		; preds = %entry
	ret void

forcond:		; preds = %entry
	%tmp3.i536 = load i32* null		; <i32> [#uses=1]
	%cmp12 = icmp slt i32 0, %tmp3.i536		; <i1> [#uses=1]
	br i1 %cmp12, label %forbody, label %afterfor

forbody:		; preds = %forcond
	%bitcast204.i104 = bitcast <4 x i32> zeroinitializer to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp78 = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> < float 1.280000e+02, float 1.280000e+02, float 1.280000e+02, float 1.280000e+02 >, <4 x float> zeroinitializer) nounwind		; <<4 x float>> [#uses=2]
	%tmp79 = call <4 x i32> @llvm.x86.sse2.cvttps2dq(<4 x float> %tmp78) nounwind		; <<4 x i32>> [#uses=1]
	%tmp80 = call <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32> %tmp79) nounwind		; <<4 x float>> [#uses=1]
	%sub140.i = sub <4 x float> %tmp78, %tmp80		; <<4 x float>> [#uses=2]
	%mul166.i = mul <4 x float> zeroinitializer, %sub140.i		; <<4 x float>> [#uses=1]
	%add167.i = add <4 x float> %mul166.i, < float 0x3FE62ACB60000000, float 0x3FE62ACB60000000, float 0x3FE62ACB60000000, float 0x3FE62ACB60000000 >		; <<4 x float>> [#uses=1]
	%mul171.i = mul <4 x float> %add167.i, %sub140.i		; <<4 x float>> [#uses=1]
	%add172.i = add <4 x float> %mul171.i, < float 0x3FF0000A40000000, float 0x3FF0000A40000000, float 0x3FF0000A40000000, float 0x3FF0000A40000000 >		; <<4 x float>> [#uses=1]
	%bitcast176.i = bitcast <4 x float> %add172.i to <4 x i32>		; <<4 x i32>> [#uses=1]
	%andnps178.i = and <4 x i32> %bitcast176.i, zeroinitializer		; <<4 x i32>> [#uses=1]
	%bitcast179.i = bitcast <4 x i32> %andnps178.i to <4 x float>		; <<4 x float>> [#uses=1]
	%mul186.i = mul <4 x float> %bitcast179.i, zeroinitializer		; <<4 x float>> [#uses=1]
	%bitcast190.i = bitcast <4 x float> %mul186.i to <4 x i32>		; <<4 x i32>> [#uses=1]
	%andnps192.i = and <4 x i32> %bitcast190.i, zeroinitializer		; <<4 x i32>> [#uses=1]
	%xorps.i = xor <4 x i32> zeroinitializer, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%orps203.i = or <4 x i32> %andnps192.i, %xorps.i		; <<4 x i32>> [#uses=1]
	%bitcast204.i = bitcast <4 x i32> %orps203.i to <4 x float>		; <<4 x float>> [#uses=1]
	%mul310 = mul <4 x float> %bitcast204.i104, zeroinitializer		; <<4 x float>> [#uses=2]
	%mul313 = mul <4 x float> %bitcast204.i, zeroinitializer		; <<4 x float>> [#uses=1]
	%cmpunord.i11 = call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> zeroinitializer, <4 x float> zeroinitializer, i8 3) nounwind		; <<4 x float>> [#uses=1]
	%bitcast6.i13 = bitcast <4 x float> %cmpunord.i11 to <4 x i32>		; <<4 x i32>> [#uses=2]
	%andps.i14 = and <4 x i32> zeroinitializer, %bitcast6.i13		; <<4 x i32>> [#uses=1]
	%not.i16 = xor <4 x i32> %bitcast6.i13, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%andnps.i17 = and <4 x i32> zeroinitializer, %not.i16		; <<4 x i32>> [#uses=1]
	%orps.i18 = or <4 x i32> %andnps.i17, %andps.i14		; <<4 x i32>> [#uses=1]
	%bitcast17.i19 = bitcast <4 x i32> %orps.i18 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp83 = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %mul310, <4 x float> zeroinitializer) nounwind		; <<4 x float>> [#uses=1]
	%bitcast.i3 = bitcast <4 x float> %mul310 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%andps.i5 = and <4 x i32> %bitcast.i3, zeroinitializer		; <<4 x i32>> [#uses=1]
	%bitcast11.i6 = bitcast <4 x float> %tmp83 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%not.i7 = xor <4 x i32> zeroinitializer, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%andnps.i8 = and <4 x i32> %bitcast11.i6, %not.i7		; <<4 x i32>> [#uses=1]
	%orps.i9 = or <4 x i32> %andnps.i8, %andps.i5		; <<4 x i32>> [#uses=1]
	%bitcast17.i10 = bitcast <4 x i32> %orps.i9 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp84 = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %mul313, <4 x float> zeroinitializer) nounwind		; <<4 x float>> [#uses=1]
	%bitcast6.i = bitcast <4 x float> zeroinitializer to <4 x i32>		; <<4 x i32>> [#uses=2]
	%andps.i = and <4 x i32> zeroinitializer, %bitcast6.i		; <<4 x i32>> [#uses=1]
	%bitcast11.i = bitcast <4 x float> %tmp84 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%not.i = xor <4 x i32> %bitcast6.i, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%andnps.i = and <4 x i32> %bitcast11.i, %not.i		; <<4 x i32>> [#uses=1]
	%orps.i = or <4 x i32> %andnps.i, %andps.i		; <<4 x i32>> [#uses=1]
	%bitcast17.i = bitcast <4 x i32> %orps.i to <4 x float>		; <<4 x float>> [#uses=1]
	call void null(<4 x float> %bitcast17.i19, <4 x float> %bitcast17.i10, <4 x float> %bitcast17.i, <4 x float> zeroinitializer, %struct.__ImageExecInfo* null, <4 x i32> zeroinitializer) nounwind
	unreachable

afterfor:		; preds = %forcond
	ret void
}

declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8) nounwind readnone

declare <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32>) nounwind readnone

declare <4 x i32> @llvm.x86.sse2.cvttps2dq(<4 x float>) nounwind readnone

declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone
