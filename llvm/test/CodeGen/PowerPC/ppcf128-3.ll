; RUN: llc -verify-machineinstrs < %s -march=ppc32
	%struct.stp_sequence = type { double, double }

define i32 @stp_sequence_set_short_data(%struct.stp_sequence* %sequence, i32 %count, i16* %data) {
entry:
	%tmp1112 = sitofp i16 0 to ppc_fp128		; <ppc_fp128> [#uses=1]
	%tmp13 = call i32 (...) @__inline_isfinite( ppc_fp128 %tmp1112 ) nounwind 		; <i32> [#uses=0]
	ret i32 0
}

define i32 @stp_sequence_set_short_data2(%struct.stp_sequence* %sequence, i32 %count, i16* %data) {
entry:
	%tmp1112 = sitofp i8 0 to ppc_fp128		; <ppc_fp128> [#uses=1]
	%tmp13 = call i32 (...) @__inline_isfinite( ppc_fp128 %tmp1112 ) nounwind 		; <i32> [#uses=0]
	ret i32 0
}

define i32 @stp_sequence_set_short_data3(%struct.stp_sequence* %sequence, i32 %count, i16* %data) {
entry:
	%tmp1112 = uitofp i16 0 to ppc_fp128		; <ppc_fp128> [#uses=1]
	%tmp13 = call i32 (...) @__inline_isfinite( ppc_fp128 %tmp1112 ) nounwind 		; <i32> [#uses=0]
	ret i32 0
}

define i32 @stp_sequence_set_short_data4(%struct.stp_sequence* %sequence, i32 %count, i16* %data) {
entry:
	%tmp1112 = uitofp i8 0 to ppc_fp128		; <ppc_fp128> [#uses=1]
	%tmp13 = call i32 (...) @__inline_isfinite( ppc_fp128 %tmp1112 ) nounwind 		; <i32> [#uses=0]
	ret i32 0
}

declare i32 @__inline_isfinite(...)
