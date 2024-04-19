; RUN: llc -mtriple=x86_64 < %s | FileCheck %s

declare void @tail_call_target()

define void @conditional_tail_call(i32 %cond) "function-instrument"="xray-always" nounwind {
  ; CHECK-LABEL: conditional_tail_call:
  ; CHECK-NEXT:  .Lfunc_begin0:
  ; CHECK-NEXT:  # %bb.0:
  ; CHECK-NEXT:    .p2align 1, 0x90
  ; CHECK-NEXT:  .Lxray_sled_0:
  ; CHECK-NEXT:    .ascii "\353\t"
  ; CHECK-NEXT:    nopw 512(%rax,%rax)
  ; CHECK-NEXT:    testl %edi, %edi
  ; CHECK-NEXT:    je .Ltmp0
  ; CHECK-NEXT:    .p2align 1, 0x90
  ; CHECK-NEXT:  .Lxray_sled_1:
  ; CHECK-NEXT:    .ascii "\353\t"
  ; CHECK-NEXT:    nopw  512(%rax,%rax)
  ; CHECK-NEXT:  .Ltmp1:
  ; CHECK-NEXT:    jmp tail_call_target@PLT # TAILCALL
  ; CHECK-NEXT:  .Ltmp0:
  ; CHECK-NEXT:  # %bb.1:
  ; CHECK-NEXT:   .p2align  1, 0x90
  ; CHECK-NEXT:  .Lxray_sled_2:
  ; CHECK-NEXT:    retq
  ; CHECK-NEXT:    nopw %cs:512(%rax,%rax)
  ; CHECK-NEXT:  .Lfunc_end0:
  %cmp = icmp ne i32 %cond, 0
  br i1 %cmp, label %docall, label %ret
docall:
  tail call void @tail_call_target()
  ret void
ret:
  ret void
}
