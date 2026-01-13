# RUN: llvm-mc -filetype=obj -triple x86_64 %s -o - \
# RUN:   | llvm-objdump -d - | FileCheck %s

## Test that instructions inside bundle-locked groups are relaxed even if their
## fixup is short enough not to warrant relaxation on its own.
  .text
relax_in_bundle:
  .bundle_align_mode 4
  pushq   %rbp

  movl    %edi, %ebx
  callq   bar
  movl    %eax, %r14d
  imull   $17, %ebx, %ebp
  movl    %ebx, %edi
  callq   bar
  cmpl    %r14d, %ebp
  .bundle_lock

  jle     .L_ELSE
## This group would've started at 0x18 and is too long, so a chunky NOP padding
## is inserted to push it to 0x20.
# CHECK: 18: {{[a-f0-9 ]+}} nopl

## The long encoding for JLE should be used here even though its target is close
# CHECK-NEXT: 20: 0f 8e

  addl    %ebp, %eax

  jmp     .L_RET
## Same for the JMP
# CHECK: 28: e9

  .bundle_unlock

.L_ELSE:
  imull   %ebx, %eax
.L_RET:

  popq    %rbx

## Test that an instruction near a bundle end gets properly padded to the next
## bundle after it is relaxed.
  .align 16
relax_at_bundle_end:
  .rept 14
  push %rax
  .endr
# CHECK: 4d: 50 pushq
# CHECK-NEXT: 4e: {{[a-f0-9 ]+}} nop
# CHECK-NEXT: 50: 0f 85
  jne 0x100

