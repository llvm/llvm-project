; RUN: llc -mtriple=x86_64-linux-gnu < %s | FileCheck %s

; A callee-pop function can pop more bytes than ret's 16-bit immediate holds. On
; 64-bit targets that is done by popping the return address, dropping the bytes
; and pushing it back, through a register that is not live.

declare void @use(ptr)

define tailcc void @big([9000 x i64] %a) {
; CHECK-LABEL: big:
; CHECK:         popq %rax
; CHECK-NEXT:    addq $71960, %rsp
; CHECK-NEXT:    pushq %rax
; CHECK-NEXT:    retq
  ret void
}

; The register holding the result is not used.
define tailcc i64 @big_ret(i64 %x, [9000 x i64] %a) {
; CHECK-LABEL: big_ret:
; CHECK:         movq %rbx, %rax
; CHECK-NEXT:    popq %rbx
; CHECK:         popq %rcx
; CHECK-NEXT:    addq $71960, %rsp
; CHECK-NEXT:    pushq %rcx
; CHECK-NEXT:    retq
  call void @use(ptr null)
  ret i64 %x
}
