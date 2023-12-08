; RUN: llc < %s -mtriple=arm-apple-darwin -regalloc=fast -optimize-regalloc=0
; PR1925

	%"struct.kc::impl_Ccode_option" = type { %"struct.kc::impl_abstract_phylum" }
	%"struct.kc::impl_ID" = type { %"struct.kc::impl_abstract_phylum", ptr, ptr, i32, ptr }
	%"struct.kc::impl_abstract_phylum" = type { ptr }
	%"struct.kc::impl_casestring__Str" = type { %"struct.kc::impl_abstract_phylum", ptr }

define ptr @_ZN2kc18f_typeofunpsubtermEPNS_15impl_unpsubtermEPNS_7impl_IDE(ptr %a_unpsubterm, ptr %a_operator) {
entry:
	%tmp8 = getelementptr %"struct.kc::impl_Ccode_option", ptr %a_unpsubterm, i32 0, i32 0, i32 0		; <ptr> [#uses=0]
	br i1 false, label %bb41, label %bb55

bb41:		; preds = %entry
	ret ptr null

bb55:		; preds = %entry
	%tmp67 = tail call i32 null( ptr null )		; <i32> [#uses=0]
	%tmp97 = tail call i32 null( ptr null )		; <i32> [#uses=0]
	ret ptr null
}
