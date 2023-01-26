; RUN: llc < %s
; PR1833

	%struct.__class_type_info_pseudo = type { %struct.__type_info_pseudo }
	%struct.__type_info_pseudo = type { ptr, ptr }
@_ZTI2e1 = external constant %struct.__class_type_info_pseudo		; <ptr> [#uses=1]

define void @_Z7ex_testv() personality ptr @__gxx_personality_v0 {
entry:
	invoke void @__cxa_throw( ptr null, ptr @_ZTI2e1, ptr null ) noreturn 
			to label %UnifiedUnreachableBlock unwind label %lpad

bb14:		; preds = %lpad
	unreachable

lpad:		; preds = %entry
        %lpad1 = landingpad { ptr, i32 }
                  catch ptr null
	invoke void @__cxa_end_catch( )
			to label %bb14 unwind label %lpad17

lpad17:		; preds = %lpad
        %lpad2 = landingpad { ptr, i32 }
                  catch ptr null
	unreachable

UnifiedUnreachableBlock:		; preds = %entry
	unreachable
}

declare void @__cxa_throw(ptr, ptr, ptr) noreturn 

declare void @__cxa_end_catch()

declare i32 @__gxx_personality_v0(...) addrspace(0)
