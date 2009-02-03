; RUN: llvm-as < %s | opt -gvnpre | llvm-dis

define void @vorbis_lsp_to_curve() {
entry:
	br i1 false, label %bb.nph, label %return

bb.nph:		; preds = %entry
	br label %bb34

bb34:		; preds = %bb34, %bb.nph
	%p.1 = phi float [ 0x3FE6A09E60000000, %bb.nph ], [ %tmp48, %bb34 ]		; <float> [#uses=1]
	%tmp44 = load float* null		; <float> [#uses=1]
	%tmp46 = sub float %tmp44, 0.000000e+00		; <float> [#uses=1]
	%tmp48 = mul float %tmp46, %p.1		; <float> [#uses=1]
	br i1 false, label %bb57, label %bb34

bb57:		; preds = %bb34
	ret void

return:		; preds = %entry
	ret void
}
