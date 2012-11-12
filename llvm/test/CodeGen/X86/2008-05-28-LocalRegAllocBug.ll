; RUN: llc < %s -mtriple=i386-apple-darwin -regalloc=fast -optimize-regalloc=0

@_ZTVN10Evaluation10GridOutputILi3EEE = external constant [5 x i32 (...)*]		; <[5 x i32 (...)*]*> [#uses=1]

declare i8* @_Znwm(i32)

declare i8* @__cxa_begin_catch(i8*) nounwind 

define i32 @main(i32 %argc, i8** %argv) {
entry:
	br i1 false, label %bb37, label %bb34

bb34:		; preds = %entry
	ret i32 1

bb37:		; preds = %entry
	%tmp12.i.i.i.i.i66 = invoke i8* @_Znwm( i32 12 )
			to label %tmp12.i.i.i.i.i.noexc65 unwind label %lpad243		; <i8*> [#uses=0]

tmp12.i.i.i.i.i.noexc65:		; preds = %bb37
	unreachable

lpad243:		; preds = %bb37
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 cleanup
	%eh_ptr244 = extractvalue { i8*, i32 } %exn, 0
	store i32 (...)** getelementptr ([5 x i32 (...)*]* @_ZTVN10Evaluation10GridOutputILi3EEE, i32 0, i32 2), i32 (...)*** null, align 8
	%tmp133 = call i8* @__cxa_begin_catch( i8* %eh_ptr244 ) nounwind 		; <i8*> [#uses=0]
	unreachable
}

declare i32 @__gxx_personality_v0(...)
