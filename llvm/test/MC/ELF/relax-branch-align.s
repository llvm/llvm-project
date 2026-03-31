# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-objdump -d --no-show-raw-insn %t.o | FileCheck %s

## In the initial all-short layout, `jmp target` has displacement -129 (1 beyond
## [-128,127]). `je far_target` relaxes from 2B to 6B, shifting `target` by
## +4B, and `.p2align 4` absorbs this growth. With fresh offsets, the true
## displacement is -125, which fits in a short jmp.
##
## The two-pass approach (relaxFragment, then layoutSection) evaluates the
## backward jmp with stale target offsets (before je relaxation shifts it),
## causing unnecessary relaxation from short (2B) to near (5B).

  .text
  .p2align 4
func:
  je far_target
target:
  pushq %rbp
  pushq %rbx
  movl %edi, %ebx
  xorl %eax, %eax
  movq 8(%rsi), %rcx
  .rept 28
  testq %rcx, %rcx
  .endr
  jmp past_loop
  .p2align 4
loop:
  movq (%r8), %r8
  testq %r8, %r8
  je after
past_loop:
  cmpq %rdi, (%r8)
  jne loop
  cmpl %esi, 8(%r8)
## The jmp is unnecessarily relaxed to near (5B). after is at 0x86 instead of
## 0x83.
# CHECK:       81: jmp 0x6 <target>
# CHECK-EMPTY:
# CHECK-NEXT: 0000000000000086 <after>:
  jmp target
after:
  popq %rbx
  popq %rbp
  ret

  .space 200
far_target:
  ret
