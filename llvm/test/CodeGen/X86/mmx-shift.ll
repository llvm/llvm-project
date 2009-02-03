; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx | grep psllq | grep 32
; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+mmx | grep psllq | grep 32
; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx | grep psrad
; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+mmx | grep psrlw

define i64 @t1(<1 x i64> %mm1) nounwind  {
entry:
	%tmp6 = tail call <1 x i64> @llvm.x86.mmx.pslli.q( <1 x i64> %mm1, i32 32 )		; <<1 x i64>> [#uses=1]
	%retval1112 = bitcast <1 x i64> %tmp6 to i64		; <i64> [#uses=1]
	ret i64 %retval1112
}

declare <1 x i64> @llvm.x86.mmx.pslli.q(<1 x i64>, i32) nounwind readnone 

define i64 @t2(<2 x i32> %mm1, <2 x i32> %mm2) nounwind  {
entry:
	%tmp7 = tail call <2 x i32> @llvm.x86.mmx.psra.d( <2 x i32> %mm1, <2 x i32> %mm2 ) nounwind readnone 		; <<2 x i32>> [#uses=1]
	%retval1112 = bitcast <2 x i32> %tmp7 to i64		; <i64> [#uses=1]
	ret i64 %retval1112
}

declare <2 x i32> @llvm.x86.mmx.psra.d(<2 x i32>, <2 x i32>) nounwind readnone 

define i64 @t3(<1 x i64> %mm1, i32 %bits) nounwind  {
entry:
	%tmp6 = bitcast <1 x i64> %mm1 to <4 x i16>		; <<4 x i16>> [#uses=1]
	%tmp8 = tail call <4 x i16> @llvm.x86.mmx.psrli.w( <4 x i16> %tmp6, i32 %bits ) nounwind readnone 		; <<4 x i16>> [#uses=1]
	%retval1314 = bitcast <4 x i16> %tmp8 to i64		; <i64> [#uses=1]
	ret i64 %retval1314
}

declare <4 x i16> @llvm.x86.mmx.psrli.w(<4 x i16>, i32) nounwind readnone 
