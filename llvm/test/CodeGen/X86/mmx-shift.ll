; RUN: llc < %s -march=x86 -mattr=+mmx | grep psllq | grep 32
; RUN: llc < %s -march=x86-64 -mattr=+mmx | grep psllq | grep 32
; RUN: llc < %s -march=x86 -mattr=+mmx | grep psrad
; RUN: llc < %s -march=x86-64 -mattr=+mmx | grep psrlw

define i64 @t1(<1 x i64> %mm1) nounwind  {
entry:
        %tmp = bitcast <1 x i64> %mm1 to x86_mmx
	%tmp6 = tail call x86_mmx @llvm.x86.mmx.pslli.q( x86_mmx %tmp, i32 32 )		; <x86_mmx> [#uses=1]
        %retval1112 = bitcast x86_mmx %tmp6 to i64
	ret i64 %retval1112
}

declare x86_mmx @llvm.x86.mmx.pslli.q(x86_mmx, i32) nounwind readnone 

define i64 @t2(x86_mmx %mm1, x86_mmx %mm2) nounwind  {
entry:
	%tmp7 = tail call x86_mmx @llvm.x86.mmx.psra.d( x86_mmx %mm1, x86_mmx %mm2 ) nounwind readnone 		; <x86_mmx> [#uses=1]
        %retval1112 = bitcast x86_mmx %tmp7 to i64
	ret i64 %retval1112
}

declare x86_mmx @llvm.x86.mmx.psra.d(x86_mmx, x86_mmx) nounwind readnone 

define i64 @t3(x86_mmx %mm1, i32 %bits) nounwind  {
entry:
	%tmp8 = tail call x86_mmx @llvm.x86.mmx.psrli.w( x86_mmx %mm1, i32 %bits ) nounwind readnone 		; <x86_mmx> [#uses=1]
        %retval1314 = bitcast x86_mmx %tmp8 to i64
	ret i64 %retval1314
}

declare x86_mmx @llvm.x86.mmx.psrli.w(x86_mmx, i32) nounwind readnone 
