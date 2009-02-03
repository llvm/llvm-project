; RUN: llvm-as < %s | llc -march=x86-64 -stats  |& \
; RUN:   grep {6 .*Number of machine instrs printed}

;; Integer absolute value, should produce something at least as good as:
;;       movl %edi, %eax
;;       sarl $31, %eax
;;       addl %eax, %edi
;;       xorl %eax, %edi
;;       movl %edi, %eax
;;       ret
define i32 @test(i32 %a) nounwind {
        %tmp1neg = sub i32 0, %a
        %b = icmp sgt i32 %a, -1
        %abs = select i1 %b, i32 %a, i32 %tmp1neg
        ret i32 %abs
}

