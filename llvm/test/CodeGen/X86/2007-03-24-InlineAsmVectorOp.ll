; RUN: llc < %s -mcpu=yonah -march=x86 | \
; RUN:   grep "cmpltsd %xmm0, %xmm0"
target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin9"


define void @acoshf() {
	%tmp19 = tail call <2 x double> asm sideeffect "pcmpeqd $0, $0 \0A\09 cmpltsd $0, $0", "=x,0,~{dirflag},~{fpsr},~{flags}"( <2 x double> zeroinitializer )		; <<2 x double>> [#uses=0]
	ret void
}

