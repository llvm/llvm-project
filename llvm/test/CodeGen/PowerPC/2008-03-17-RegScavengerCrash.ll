; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--

	%struct._cpp_strbuf = type { ptr, i32, i32 }
	%struct.cpp_string = type { i32, ptr }

declare fastcc void @emit_numeric_escape(i32, i32, ptr, i32) nounwind 

define i32 @cpp_interpret_string(i32 %pfile, ptr %from, i32 %wide) nounwind  {
entry:
	%tmp61 = load i32, ptr null, align 4		; <i32> [#uses=1]
	%toBool = icmp eq i32 %wide, 0		; <i1> [#uses=2]
	%iftmp.87.0 = select i1 %toBool, i32 %tmp61, i32 0		; <i32> [#uses=2]
	%tmp69 = icmp ult i32 %iftmp.87.0, 33		; <i1> [#uses=1]
	%min = select i1 %tmp69, i32 %iftmp.87.0, i32 32		; <i32> [#uses=1]
	%tmp71 = icmp ugt i32 %min, 31		; <i1> [#uses=1]
	br i1 %tmp71, label %bb79, label %bb75
bb75:		; preds = %entry
	ret i32 0
bb79:		; preds = %entry
	br i1 %toBool, label %bb103, label %bb94
bb94:		; preds = %bb79
	br i1 false, label %bb729, label %bb130.preheader
bb103:		; preds = %bb79
	ret i32 0
bb130.preheader:		; preds = %bb94
	%tmp134 = getelementptr %struct.cpp_string, ptr %from, i32 0, i32 1		; <ptr> [#uses=0]
	ret i32 0
bb729:		; preds = %bb94
	call fastcc void @emit_numeric_escape( i32 %pfile, i32 0, ptr null, i32 %wide ) nounwind 
	ret i32 1
}
