; RUN: llc < %s -mtriple=arm-apple-darwin -regalloc=fast -optimize-regalloc=0
; PR1925

	%"struct.kc::impl_Ccode_option" = type { %"struct.kc::impl_abstract_phylum" }
	%"struct.kc::impl_ID" = type { %"struct.kc::impl_abstract_phylum", %"struct.kc::impl_Ccode_option"*, %"struct.kc::impl_casestring__Str"*, i32, %"struct.kc::impl_casestring__Str"* }
	%"struct.kc::impl_abstract_phylum" = type { i32 (...)** }
	%"struct.kc::impl_casestring__Str" = type { %"struct.kc::impl_abstract_phylum", i8* }

define %"struct.kc::impl_ID"* @_ZN2kc18f_typeofunpsubtermEPNS_15impl_unpsubtermEPNS_7impl_IDE(%"struct.kc::impl_Ccode_option"* %a_unpsubterm, %"struct.kc::impl_ID"* %a_operator) {
entry:
	%tmp8 = getelementptr %"struct.kc::impl_Ccode_option"* %a_unpsubterm, i32 0, i32 0, i32 0		; <i32 (...)***> [#uses=0]
	br i1 false, label %bb41, label %bb55

bb41:		; preds = %entry
	ret %"struct.kc::impl_ID"* null

bb55:		; preds = %entry
	%tmp67 = tail call i32 null( %"struct.kc::impl_abstract_phylum"* null )		; <i32> [#uses=0]
	%tmp97 = tail call i32 null( %"struct.kc::impl_abstract_phylum"* null )		; <i32> [#uses=0]
	ret %"struct.kc::impl_ID"* null
}
