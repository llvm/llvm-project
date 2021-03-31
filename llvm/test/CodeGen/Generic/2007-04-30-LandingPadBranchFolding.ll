; RUN: llc < %s
; PR1228

	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Alloc_hider" = type { i8* }
	%"struct.std::locale" = type { %"struct.std::locale::_Impl"* }
	%"struct.std::locale::_Impl" = type { i32, %"struct.std::locale::facet"**, i32, %"struct.std::locale::facet"**, i8** }
	%"struct.std::locale::facet" = type { i32 (...)**, i32 }
	%"struct.std::string" = type { %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Alloc_hider" }

define void @_ZNKSt6locale4nameEv(%"struct.std::string"* %agg.result) personality i32 (...)* @__gxx_personality_v0 {
entry:
	%tmp105 = icmp eq i8* null, null		; <i1> [#uses=1]
	br i1 %tmp105, label %cond_true, label %cond_true222

cond_true:		; preds = %entry
	invoke void @_ZNSs14_M_replace_auxEjjjc( )
			to label %cond_next1328 unwind label %cond_true1402

cond_true222:		; preds = %cond_true222, %entry
	%tmp207 = call i32 @strcmp( )		; <i32> [#uses=1]
	%tmp208 = icmp eq i32 %tmp207, 0		; <i1> [#uses=2]
	%bothcond1480 = and i1 %tmp208, false		; <i1> [#uses=1]
	br i1 %bothcond1480, label %cond_true222, label %cond_next226.loopexit

cond_next226.loopexit:		; preds = %cond_true222
	%phitmp = xor i1 %tmp208, true		; <i1> [#uses=1]
	br i1 %phitmp, label %cond_false280, label %cond_true235

cond_true235:		; preds = %cond_next226.loopexit
	invoke void @_ZNSs6assignEPKcj( )
			to label %cond_next1328 unwind label %cond_true1402

cond_false280:		; preds = %cond_next226.loopexit
	invoke void @_ZNSs7reserveEj( )
			to label %invcont282 unwind label %cond_true1402

invcont282:		; preds = %cond_false280
	invoke void @_ZNSs6appendEPKcj( )
			to label %invcont317 unwind label %cond_true1402

invcont317:		; preds = %invcont282
	ret void

cond_next1328:		; preds = %cond_true235, %cond_true
	ret void

cond_true1402:		; preds = %invcont282, %cond_false280, %cond_true235, %cond_true
  %lpad = landingpad { i8*, i32 }
            cleanup
  ret void
}

declare void @_ZNSs14_M_replace_auxEjjjc()

declare i32 @strcmp()

declare void @_ZNSs6assignEPKcj()

declare void @_ZNSs7reserveEj()

declare void @_ZNSs6appendEPKcj()

declare i32 @__gxx_personality_v0(...) addrspace(0)
