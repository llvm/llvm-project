; REQUIRES: asserts
; RUN: llc < %s -march=x86 -mattr=+sse2 -mcpu=penryn -stats 2>&1 | \
; RUN:   grep "twoaddressinstruction" | grep "Number of instructions aggressively commuted"
; rdar://6480363

target triple = "i386-apple-darwin9.6"

define <2 x double> @t(<2 x double> %A, <2 x double> %B, <2 x double> %C) nounwind readnone {
entry:
	%tmp.i3 = bitcast <2 x double> %B to <2 x i64>		; <<2 x i64>> [#uses=1]
	%tmp2.i = or <2 x i64> %tmp.i3, <i64 4607632778762754458, i64 4607632778762754458>		; <<2 x i64>> [#uses=1]
	%tmp3.i = bitcast <2 x i64> %tmp2.i to <2 x double>		; <<2 x double>> [#uses=1]
	%tmp.i2 = fadd <2 x double> %tmp3.i, %A		; <<2 x double>> [#uses=1]
	%tmp.i = fadd <2 x double> %tmp.i2, %C		; <<2 x double>> [#uses=1]
	ret <2 x double> %tmp.i
}
