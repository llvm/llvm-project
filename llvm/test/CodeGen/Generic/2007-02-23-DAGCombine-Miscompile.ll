; PR1219
; RUN: llvm-as < %s | llc -march=x86 | grep {movl	\$1, %eax}

define i32 @test(i1 %X) {
old_entry1:
        %hvar2 = zext i1 %X to i32
	%C = icmp sgt i32 %hvar2, -1
	br i1 %C, label %cond_true15, label %cond_true
cond_true15:
        ret i32 1
cond_true:
        ret i32 2
}
