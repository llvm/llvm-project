; RUN: llvm-as < %s | llc -march=x86 | not grep 120
; Don't accidentally add the offset twice for trailing bytes.

	%struct.S63 = type { [63 x i8] }
@g1s63 = external global %struct.S63		; <%struct.S63*> [#uses=1]

declare void @test63(%struct.S63* byval align 4 ) nounwind 

define void @testit63_entry_2E_ce() nounwind  {
	tail call void @test63( %struct.S63* byval align 4  @g1s63 ) nounwind 
	ret void
}
