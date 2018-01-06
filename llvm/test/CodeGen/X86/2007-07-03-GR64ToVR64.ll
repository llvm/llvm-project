; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+mmx | FileCheck %s

; CHECK: movq %rsi, [[MM0:%mm[0-9]+]]
; CHECK: movq %rdi, [[MM1:%mm[0-9]+]]
; CHECK: paddusw [[MM0]], [[MM1]]

@R = external global x86_mmx		; <x86_mmx*> [#uses=1]

define void @foo(<1 x i64> %A, <1 x i64> %B) nounwind {
entry:
	%tmp4 = bitcast <1 x i64> %B to x86_mmx		; <<4 x i16>> [#uses=1]
	%tmp6 = bitcast <1 x i64> %A to x86_mmx		; <<4 x i16>> [#uses=1]
	%tmp7 = tail call x86_mmx @llvm.x86.mmx.paddus.w( x86_mmx %tmp6, x86_mmx %tmp4 )		; <x86_mmx> [#uses=1]
	store x86_mmx %tmp7, x86_mmx* @R
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

declare x86_mmx @llvm.x86.mmx.paddus.w(x86_mmx, x86_mmx)
declare void @llvm.x86.mmx.emms()
