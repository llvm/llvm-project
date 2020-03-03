  # RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu --x86-align-branch-boundary=32 --x86-align-branch=jmp %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

  # Exercise cases where we're enabling interrupts with one instruction delay
  # and thus can't add a nop in between without changing behavior.

  .text

  # CHECK: 1e:       sti
  # CHECK: 1f:       jmp
  .p2align  5
  .rept 30
  int3
  .endr
  sti
  jmp baz

  # CHECK: 5c:       movq %rax, %ss
  # CHECK: 5f:       jmp
  .p2align  5
  .rept 28
  int3
  .endr
  movq %rax, %ss
  jmp baz

  # CHECK: 9d:       movl %esi, %ss
  # CHECK: 9f:       jmp
  .p2align  5
  .rept 29
  int3
  .endr
  movl %esi, %ss
  jmp baz

  # movw and movl are interchangeable since we're only using the low 16 bits.
  # Both are generated as "MOV Sreg,r/m16**" (8E /r), but disassembled as movl
  # CHECK: dd:       movl %esi, %ss
  # CHECK: df:       jmp
  .p2align  5
  .rept 29
  int3
  .endr
  movw %si, %ss
  jmp baz

  # CHECK: 11b:       movw (%esi), %ss
  # CHECK: 11e:       jmp
  .p2align  5
  .rept 27
  int3
  .endr
  movw (%esi), %ss
  jmp baz

  # CHECK: 15b:      	movw	(%rsi), %ss
  # CHECK: 15d:     	jmp
  .p2align  5
  .rept 27
  int3
  .endr
  movw (%rsi), %ss
  jmp baz


  int3
  .section ".text.other"
bar:
  retq
