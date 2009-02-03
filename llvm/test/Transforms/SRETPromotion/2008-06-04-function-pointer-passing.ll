; This test lures sretpromotion into promoting the sret argument of foo, even
; when the function is used as an argument to bar. It used to not check for
; this, assuming that all users of foo were direct calls, resulting in an
; assertion failure later on.

; We're mainly testing for opt not to crash, but we'll check to see if the sret
; attribute is still there for good measure.
; RUN: llvm-as < %s | opt -sretpromotion | llvm-dis | grep sret

%struct.S = type <{ i32, i32 }>

define i32 @main() {
entry:
	%tmp = alloca %struct.S		; <%struct.S*> [#uses=1]
	call void @bar( %struct.S* sret  %tmp, void (%struct.S*, ...)* @foo )
	ret i32 undef
}

declare void @bar(%struct.S* sret , void (%struct.S*, ...)*)

define internal void @foo(%struct.S* sret  %agg.result, ...) {
entry:
	ret void
}
