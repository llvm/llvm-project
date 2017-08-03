; REQUIRES: asserts
; RUN: llc < %s -march=x86 -mattr=+sse2 -stats 2>&1 | FileCheck %s
; rdar://6608609

; CHECK-NOT: commuted

define <2 x double> @t(<2 x double> %A, <2 x double> %B, <2 x double> %C) nounwind readnone {
entry:
	%tmp.i2 = bitcast <2 x double> %B to <2 x i64>		; <<2 x i64>> [#uses=1]
	%tmp2.i = or <2 x i64> %tmp.i2, <i64 4607632778762754458, i64 4607632778762754458>		; <<2 x i64>> [#uses=1]
	%tmp3.i = bitcast <2 x i64> %tmp2.i to <2 x double>		; <<2 x double>> [#uses=1]
	%0 = tail call <2 x double> @llvm.x86.sse2.add.sd(<2 x double> %A, <2 x double> %tmp3.i) nounwind readnone		; <<2 x double>> [#uses=1]
	%tmp.i = fadd <2 x double> %0, %C		; <<2 x double>> [#uses=1]
	ret <2 x double> %tmp.i
}

declare <2 x double> @llvm.x86.sse2.add.sd(<2 x double>, <2 x double>) nounwind readnone
