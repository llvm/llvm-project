; RUN: llc < %s -march=x86-64 | FileCheck %s

;; Integer absolute value, should produce something at least as good as:
;;       movl   %edi, %eax
;;       negl   %eax
;;       cmovll %edi, %eax
;;       ret
; rdar://10695237
define i32 @test(i32 %a) nounwind {
; CHECK: test:
; CHECK: mov
; CHECK-NEXT: neg
; CHECK-NEXT: cmov
; CHECK-NEXT: ret
        %tmp1neg = sub i32 0, %a
        %b = icmp sgt i32 %a, -1
        %abs = select i1 %b, i32 %a, i32 %tmp1neg
        ret i32 %abs
}

