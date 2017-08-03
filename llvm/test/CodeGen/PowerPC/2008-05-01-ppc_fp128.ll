; RUN: llc -verify-machineinstrs < %s
target triple = "powerpc-apple-darwin9.2.2"

define i256 @func(ppc_fp128 %a, ppc_fp128 %b, ppc_fp128 %c, ppc_fp128 %d) nounwind readnone  {
entry:
	br i1 false, label %bb36, label %bb484

bb36:		; preds = %entry
	%tmp124 = fcmp ord ppc_fp128 %b, 0xM00000000000000000000000000000000		; <i1> [#uses=1]
	%tmp140 = and i1 %tmp124, fcmp une (ppc_fp128 0xM00000000000000000000000000000000, ppc_fp128 0xM00000000000000000000000000000000)		; <i1> [#uses=0]
	unreachable

bb484:		; preds = %entry
	ret i256 0
}
