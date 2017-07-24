; RUN: llc < %s -mtriple=i386-linux-gnu  | FileCheck %s

declare x86_regcallcc i32 @callee(i32 %a0, i32 %b0, i32 %c0, i32 %d0, i32 %e0);

; In RegCall calling convention, ESI and EDI are callee saved registers.
; One might think that the caller could assume that ESI value is the same before
; and after calling the callee.
; However, RegCall also says that a register that was used for 
; passing/returning argumnets, can be assumed to be modified by the callee.
; In other words, it is no longer a callee saved register.
; In this case we want to see that EDX/ECX values are saved and EDI/ESI are assumed
; to be modified by the callee.
; This is a hipe CC function that doesn't save any register for the caller.
; So we can be sure that there is no other reason to save EDX/ECX.
; The caller arguments are expected to be passed (in the following order) 
; in registers: ESI, EBP, EAX, EDX and ECX.
define cc 11 i32 @caller(i32 %a0, i32 %b0, i32 %c0, i32 %d0, i32 %e0) nounwind {
  %b1 = call x86_regcallcc i32 @callee(i32 %a0, i32 %b0, i32 %c0, i32 %d0, i32 %e0)
  %b2 = add i32 %b1, %d0
  %b3 = add i32 %b2, %e0
  ret i32 %b3
}
; CHECK-LABEL:  caller
; CHECK:        subl    $12, %esp
; CHECK-NEXT:   movl    %ecx, 8(%esp)
; CHECK-NEXT:   movl    %edx, %ebx
; CHECK-NEXT:   movl    %eax, %edx
; CHECK-NEXT:   movl    %esi, %eax
; CHECK-NEXT:   movl    %ebp, %ecx
; CHECK-NEXT:   movl    %ebx, %edi
; CHECK-NEXT:   movl    8(%esp), %ebp
; CHECK-NEXT:   movl    %ebp, %esi
; CHECK-NEXT:   calll   callee
; CHECK-NEXT:   leal    (%eax,%ebx), %esi
; CHECK-NEXT:   addl    %ebp, %esi
; CHECK-NEXT:   addl    $12, %esp
; CHECK-NEXT:   retl

!hipe.literals = !{ !0, !1, !2 }
!0 = !{ !"P_NSP_LIMIT", i32 120 }
!1 = !{ !"X86_LEAF_WORDS", i32 24 }
!2 = !{ !"AMD64_LEAF_WORDS", i32 18 }

; Make sure that the callee doesn't save parameters that were passed as arguments.
; The caller arguments are expected to be passed (in the following order) 
; in registers: EAX, ECX, EDX, EDI and ESI.
; The result will return in EAX, ECX and EDX.
define x86_regcallcc {i32, i32, i32} @test_callee(i32 %a0, i32 %b0, i32 %c0, i32 %d0, i32 %e0) nounwind {
  %b1 = mul i32 7, %e0
  %b2 = udiv i32 5, %e0
  %b3 = mul i32 7, %d0
  %b4 = insertvalue {i32, i32, i32} undef, i32 %b1, 0
  %b5 = insertvalue {i32, i32, i32} %b4, i32 %b2, 1
  %b6 = insertvalue {i32, i32, i32} %b5, i32 %b3, 2
  ret {i32, i32, i32} %b6
}
; CHECK-LABEL: test_callee
; CHECK-NOT:   pushl %esi
; CHECK-NOT:   pushl %edi
; CHECK:       retl
