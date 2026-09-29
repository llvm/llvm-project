; RUN: opt < %s -passes=instcombine -disable-output
; PR2303
	%"struct.std::ctype<char>" = type { %"struct.std::locale::facet", ptr, i8, ptr, ptr, ptr, i8, [256 x i8], [256 x i8], i8 }
	%"struct.std::locale::facet" = type { ptr, i32 }

declare ptr @_ZNSt6locale5facet15_S_get_c_localeEv()

declare ptr @__ctype_toupper_loc() readnone 

declare ptr @__ctype_tolower_loc() readnone 

define void @_ZNSt5ctypeIcEC2EPiPKtbm(ptr %this, ptr %unnamed_arg, ptr %__table, i8 zeroext  %__del, i64 %__refs) personality ptr @__gxx_personality_v0 {
entry:
	%tmp8 = invoke ptr @_ZNSt6locale5facet15_S_get_c_localeEv( )
			to label %invcont unwind label %lpad		; <ptr> [#uses=0]

invcont:		; preds = %entry
	%tmp32 = invoke ptr @__ctype_toupper_loc( ) readnone 
			to label %invcont31 unwind label %lpad		; <ptr> [#uses=0]

invcont31:		; preds = %invcont
	%tmp38 = invoke ptr @__ctype_tolower_loc( ) readnone 
			to label %invcont37 unwind label %lpad		; <ptr> [#uses=1]

invcont37:		; preds = %invcont31
	%tmp39 = load ptr, ptr %tmp38, align 8		; <ptr> [#uses=1]
	%tmp41 = getelementptr %"struct.std::ctype<char>", ptr %this, i32 0, i32 4		; <ptr> [#uses=1]
	store ptr %tmp39, ptr %tmp41, align 8
	ret void

lpad:		; preds = %invcont31, %invcont, %entry
        %exn = landingpad {ptr, i32}
                 cleanup
	unreachable
}

declare i32 @__gxx_personality_v0(...)
