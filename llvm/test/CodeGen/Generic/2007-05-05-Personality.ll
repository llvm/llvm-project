; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -enable-eh -o - | grep zPLR

@error = external global i8		; <i8*> [#uses=2]

define void @_ada_x() {
entry:
	invoke void @raise( )
			to label %eh_then unwind label %unwind

unwind:		; preds = %entry
	%eh_ptr = tail call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select = tail call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i8* @error )		; <i32> [#uses=1]
	%eh_typeid = tail call i32 @llvm.eh.typeid.for.i32( i8* @error )		; <i32> [#uses=1]
	%tmp2 = icmp eq i32 %eh_select, %eh_typeid		; <i1> [#uses=1]
	br i1 %tmp2, label %eh_then, label %Unwind

eh_then:		; preds = %unwind, %entry
	ret void

Unwind:		; preds = %unwind
	tail call i32 (...)* @_Unwind_Resume( i8* %eh_ptr )		; <i32>:0 [#uses=0]
	unreachable
}

declare void @raise()

declare i8* @llvm.eh.exception()

declare i32 @llvm.eh.selector.i32(i8*, i8*, ...)

declare i32 @llvm.eh.typeid.for.i32(i8*)

declare i32 @__gnat_eh_personality(...)

declare i32 @_Unwind_Resume(...)
