; RUN: llc < %s -mtriple=i386-apple-darwin -regalloc=fast -optimize-regalloc=0

@_ZTVN10Evaluation10GridOutputILi3EEE = external constant [5 x ptr]		; <ptr> [#uses=1]

declare ptr @_Znwm(i32)

declare ptr @__cxa_begin_catch(ptr) nounwind 

define i32 @main(i32 %argc, ptr %argv) personality ptr @__gxx_personality_v0 {
entry:
	br i1 false, label %bb37, label %bb34

bb34:		; preds = %entry
	ret i32 1

bb37:		; preds = %entry
	%tmp12.i.i.i.i.i66 = invoke ptr @_Znwm( i32 12 )
			to label %tmp12.i.i.i.i.i.noexc65 unwind label %lpad243		; <ptr> [#uses=0]

tmp12.i.i.i.i.i.noexc65:		; preds = %bb37
	unreachable

lpad243:		; preds = %bb37
        %exn = landingpad {ptr, i32}
                 cleanup
	%eh_ptr244 = extractvalue { ptr, i32 } %exn, 0
	store ptr getelementptr ([5 x ptr], ptr @_ZTVN10Evaluation10GridOutputILi3EEE, i32 0, i32 2), ptr null, align 8
	%tmp133 = call ptr @__cxa_begin_catch( ptr %eh_ptr244 ) nounwind 		; <ptr> [#uses=0]
	unreachable
}

declare i32 @__gxx_personality_v0(...)
