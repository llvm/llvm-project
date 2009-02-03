; This test shows an sret function that is used as an operand to a bitcast.
; StructRetPromotion used to assume that a function was only used by call or
; invoke instructions, making this code cause an assertion failure.

; We're mainly testing for opt not to crash, but we'll check to see if the sret
; attribute is still there for good measure.
; RUN: llvm-as < %s | opt -sretpromotion | llvm-dis | grep sret

%struct.S = type <{ i32, i32 }>

define i32 @main() {
entry:
        %bar = bitcast void (%struct.S*)* @foo to i32 ()*
	ret i32 undef
}

define internal void @foo(%struct.S* sret) {
entry:
	ret void
}
