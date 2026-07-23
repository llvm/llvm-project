; RUN: llc < %s -mtriple=x86_64-linux-gnu -code-model=large -enable-misched=false | FileCheck %s --check-prefixes=CHECK,JMP
; RUN: llc < %s -mtriple=x86_64-linux-gnu -code-model=large -enable-misched=false -mattr=jmpabs | FileCheck %s --check-prefixes=CHECK,JMPABS
; RUN: llc < %s -mtriple=x86_64-linux-gnu -code-model=large -enable-misched=false -relocation-model=pic | FileCheck %s --check-prefixes=CHECK,PIC
; RUN: llc < %s -mtriple=x86_64-linux-gnu -code-model=large -enable-misched=false -mattr=jmpabs -relocation-model=pic | FileCheck %s --check-prefixes=CHECK,PIC

declare tailcc i32 @callee(i32 %arg)
define tailcc i32 @directcall(i32 %arg) {
entry:
; This is the large code model, so &callee may not fit into the jmp
; instruction.  Instead, stick it into a register.
;  JMP: movabsq $callee, [[REGISTER:%r[a-z0-9]+]]
;  JMP: jmpq    *[[REGISTER]]  # TAILCALL
;  JMPABS-NOT: movabsq
;  JMPABS:     jmpabs $callee  # TAILCALL
;  PIC: movabsq $_GLOBAL_OFFSET_TABLE_
;  PIC: movabsq $callee@GOT
;  PIC: jmpq    *
  %res = tail call tailcc i32 @callee(i32 %arg)
  ret i32 %res
}

; Check that the register used for an indirect tail call doesn't
; clobber any of the arguments.
define tailcc i32 @indirect_manyargs(ptr %target) {
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
;  PIC: movabsq $_GLOBAL_OFFSET_TABLE
;  CHECK: movl $7, 16(%rsp)
; This is the large code model, so &manyargs_callee may not fit into
; the jmp instruction.  Put it into a register which won't be clobbered
; while restoring callee-saved registers and won't be used for passing
; arguments.
;  JMP: movabsq $manyargs_callee, %rax
;  JMPABS-NOT: movabsq
;  PIC: movabsq $manyargs_callee@GOT
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
;  JMP: jmpq *%rax  # TAILCALL
;  JMPABS: jmpabs $manyargs_callee  # TAILCALL
;  PIC: jmpq    *
  %res = tail call tailcc i32 @manyargs_callee(i32 1, i32 2, i32 3, i32 4,
                                               i32 5, i32 6, i32 7)
  ret i32 %res
}
