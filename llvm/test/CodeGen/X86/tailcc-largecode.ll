; RUN: llc < %s -mtriple=x86_64-linux-gnu -code-model=large -enable-misched=false | FileCheck %s

declare tailcc i32 @callee(i32 %arg)
define tailcc i32 @directcall(i32 %arg) {
entry:
; This is the large code model, so &callee may not fit into the jmp
; instruction.  Instead, stick it into a register.
;  CHECK: movabsq $callee, [[REGISTER:%r[a-z0-9]+]]
;  CHECK: jmpq    *[[REGISTER]]  # TAILCALL
  %res = tail call tailcc i32 @callee(i32 %arg)
  ret i32 %res
}

; Check that the register used for an indirect tail call doesn't
; clobber any of the arguments.
define tailcc i32 @indirect_manyargs(i32(i32,i32,i32,i32,i32,i32,i32)* %target) {
; Adjust the stack to enter the function.  (The amount of the
; adjustment may change in the future, in which case the location of
; the stack argument and the return adjustment will change too.)
;  CHECK: pushq
; Put the call target into R11, which won't be clobbered while restoring
; callee-saved registers and won't be used for passing arguments.
;  CHECK: movq %rdi, %rax
; Pass the stack argument.
;  CHECK: movl $7, 16(%rsp)
; Pass the register arguments, in the right registers.
;  CHECK: movl $1, %edi
;  CHECK: movl $2, %esi
;  CHECK: movl $3, %edx
;  CHECK: movl $4, %ecx
;  CHECK: movl $5, %r8d
;  CHECK: movl $6, %r9d
; Adjust the stack to "return".
;  CHECK: popq
; And tail-call to the target.
;  CHECK: jmpq *%rax  # TAILCALL
  %res = tail call tailcc i32 %target(i32 1, i32 2, i32 3, i32 4, i32 5,
                                      i32 6, i32 7)
  ret i32 %res
}

; Check that the register used for a direct tail call doesn't clobber
; any of the arguments.
declare tailcc i32 @manyargs_callee(i32,i32,i32,i32,i32,i32,i32)
define tailcc i32 @direct_manyargs() {
; Adjust the stack to enter the function.  (The amount of the
; adjustment may change in the future, in which case the location of
; the stack argument and the return adjustment will change too.)
;  CHECK: pushq
; Pass the stack argument.
;  CHECK: movl $7, 16(%rsp)
; This is the large code model, so &manyargs_callee may not fit into
; the jmp instruction.  Put it into a register which won't be clobbered
; while restoring callee-saved registers and won't be used for passing
; arguments.
;  CHECK: movabsq $manyargs_callee, %rax
; Pass the register arguments, in the right registers.
;  CHECK: movl $1, %edi
;  CHECK: movl $2, %esi
;  CHECK: movl $3, %edx
;  CHECK: movl $4, %ecx
;  CHECK: movl $5, %r8d
;  CHECK: movl $6, %r9d
; Adjust the stack to "return".
;  CHECK: popq
; And tail-call to the target.
;  CHECK: jmpq *%rax  # TAILCALL
  %res = tail call tailcc i32 @manyargs_callee(i32 1, i32 2, i32 3, i32 4,
                                               i32 5, i32 6, i32 7)
  ret i32 %res
}
