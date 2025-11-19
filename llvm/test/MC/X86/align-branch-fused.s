# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=fused+jcc foo.s | llvm-objdump -d --no-show-raw-insn - | FileCheck foo.s
# RUN: llvm-mc -filetype=obj -triple x86_64 --x86-align-branch-boundary=32 --x86-align-branch=fused+jcc -x86-pad-max-prefix-size=1 bar.s | llvm-objdump -d --no-show-raw-insn - | FileCheck bar.s

  # Exercise cases where fused instructions need to be aligned.

#--- foo.s
  .text
  .globl  foo
foo:
  .p2align  5
  .rept 30
  int3
  .endr
  # 'cmp  %rax, %rbp' is macro fused with 'jne foo',
  # so we need to align the pair.
  # CHECK:    20:          cmpq    %rax, %rbp
  # CHECK:    23:          jne
  cmp  %rax, %rbp
  jne foo
  int3

  .p2align  5
  .rept 28
  int3
  .endr
  # 'cmp  %rax, %rbp' is fusible but can not fused with `jo foo`,
  # so we only need to align 'jo foo'.
  # CHECK:    5c:          cmpq    %rax, %rbp
  cmp  %rax, %rbp
  # CHECK:    60:          jo
  jo foo
  int3

  .p2align  5
  .rept 26
  int3
  .endr
  # The second 'cmp  %rax, %rbp' is macro fused with 'jne foo'.
  cmp  %rax, %rbp
  # CHECK:    a0:          cmpq    %rax, %rbp
  # CHECK:    a3:          jne
  cmp  %rax, %rbp
  jne foo
  int3

# Exercise the case where fused instructions need to be aligned,
# ensuring fusion is not broken by a NOP

#--- bar.s
  .text
	.globl	bar
bar:
  .nops 27
# CHECK:      20:       testq   %rcx, %rcx
# CHECK:      23:       je
	testq	%rcx, %rcx
	je	.EXIT
.EXIT:
	ret
