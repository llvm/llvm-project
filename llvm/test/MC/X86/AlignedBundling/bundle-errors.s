# RUN: split-file %s %t
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/lock-without-mode.s     2>&1 | FileCheck %t/lock-without-mode.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/mode-without-arg.s      2>&1 | FileCheck %t/mode-without-arg.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/unlock-without-lock.s   2>&1 | FileCheck %t/unlock-without-lock.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/bad-lock-option.s       2>&1 | FileCheck %t/bad-lock-option.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/switch-section-locked.s 2>&1 | FileCheck %t/switch-section-locked.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/group-too-large.s       2>&1 | FileCheck %t/group-too-large.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 -mc-relax-all %t/group-too-large.s 2>&1 | FileCheck %t/group-too-large.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/nested-lock.s           2>&1 | FileCheck %t/nested-lock.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/mode-bad-size.s         2>&1 | FileCheck %t/mode-bad-size.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/align-in-lock.s         2>&1 | FileCheck %t/align-in-lock.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/unterminated-lock.s     2>&1 | FileCheck %t/unterminated-lock.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/lock-in-data.s          2>&1 | FileCheck %t/lock-in-data.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/prefix-ends-lock.s      2>&1 | FileCheck %t/prefix-ends-lock.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 -x86-branches-within-32B-boundaries %t/bundle-with-align-branch.s 2>&1 | FileCheck %t/bundle-with-align-branch.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 -x86-align-branch-boundary=32 -x86-align-branch=jmp %t/bundle-with-align-branch.s 2>&1 | FileCheck %t/bundle-with-align-branch.s

## Formats and targets without a bundling implementation reject the directives.
# RUN: not llvm-mc -filetype=obj -triple x86_64-apple-darwin %t/unsupported.s 2>&1 | FileCheck %t/unsupported.s --check-prefix=FORMAT
# RUN: not llvm-mc -filetype=obj -triple x86_64-windows-msvc %t/unsupported.s 2>&1 | FileCheck %t/unsupported.s --check-prefix=FORMAT
# RUN: %if aarch64-registered-target %{ not llvm-mc -filetype=obj -triple aarch64 %t/unsupported.s 2>&1 | FileCheck %t/unsupported.s --check-prefix=TARGET %}


#--- lock-without-mode.s
## .bundle_lock can't come without a .bundle_align_mode before it
  imull $17, %ebx, %ebp
# CHECK: [[#@LINE+1]]:3: error: .bundle_lock forbidden when bundling is disabled
  .bundle_lock
# CHECK: [[#@LINE+1]]:3: error: .bundle_unlock forbidden when bundling is disabled
  .bundle_unlock

#--- mode-without-arg.s
## .bundle_align_mode needs a following integer value
# CHECK: [[#@LINE+1]]:21: error: unknown token in expression
  .bundle_align_mode
  imull $17, %ebx, %ebp

#--- unlock-without-lock.s
## .bundle_unlock can't come without a .bundle_lock before it
  .bundle_align_mode 3
  imull $17, %ebx, %ebp
# CHECK: [[#@LINE+1]]:3: error: .bundle_unlock without matching lock
  .bundle_unlock

#--- bad-lock-option.s
## .bundle_lock can only take one `align_to_end` flag or no flag.
  .bundle_align_mode 4
# CHECK: [[#@LINE+1]]:16: error: invalid option for `.bundle_lock`
  .bundle_lock 5
  imull $17, %ebx, %ebp
  .bundle_unlock

#--- switch-section-locked.s
## This test invokes .bundle_lock and then switches to a different section
## w/o the appropriate unlock.
  .bundle_align_mode 3
  .section text1, "x"
  imull $17, %ebx, %ebp
  .bundle_lock
  imull $17, %ebx, %ebp

# CHECK: [[#@LINE+1]]:3: error: unterminated .bundle_lock
  .section text2, "x"
  imull $17, %ebx, %ebp

#--- group-too-large.s
## bundle lock size cannot be bigger than the align mode size
  .text
foo:
  .bundle_align_mode 4
  pushq   %rbp

  .bundle_lock
  pushq   %r14
  callq   bar
  callq   bar
  callq   bar
  callq   bar
# CHECK: [[#@LINE+1]]:3: error: .bundle_lock group is larger than the bundle size
  .bundle_unlock

#--- nested-lock.s
## test that nested lock is emitting the right error.
  .bundle_align_mode 4
foo:
## repeating .bundle_align_mode with the same value is allowed.
  .bundle_align_mode 4
# CHECK: [[#@LINE+1]]:3: error: .bundle_align_mode cannot be changed once set
  .bundle_align_mode 5
  .bundle_lock
# CHECK: [[#@LINE+1]]:3: error: nested .bundle_lock is not allowed
  .bundle_lock
  .bundle_unlock
  .bundle_unlock

#--- mode-bad-size.s
## Unlike GNU as, `.bundle_align_mode 0` (disabling bundling) is not supported.
# CHECK: [[#@LINE+1]]:22: error: invalid bundle alignment size (expected between 1 and 30)
  .bundle_align_mode 0
# CHECK: [[#@LINE+1]]:22: error: invalid bundle alignment size (expected between 1 and 30)
  .bundle_align_mode 31
  imull $17, %ebx, %ebp

#--- align-in-lock.s
  .bundle_align_mode 4
  .bundle_lock
  incl %eax
  .p2align 3
  incl %eax
# CHECK: [[#@LINE+1]]:3: error: alignment and .org directives are not supported inside a .bundle_lock group
  .bundle_unlock

#--- unterminated-lock.s
## A .bundle_lock left open at end of file is diagnosed rather than ignored.
  .bundle_align_mode 4
  int3
  .bundle_lock
# CHECK: [[#@LINE+1]]:3: error: unterminated .bundle_lock
  callq bar

#--- lock-in-data.s
## Padding a group is only meaningful where nops are instructions.
  .bundle_align_mode 4
  .data
  .byte 1
# CHECK: [[#@LINE+1]]:3: error: .bundle_lock is only allowed in an executable section
  .bundle_lock
  .quad 0
# CHECK: [[#@LINE+1]]:3: error: .bundle_unlock without matching lock
  .bundle_unlock

#--- prefix-ends-lock.s
## A group ending in a prefix cannot keep it attached to its instruction.
  .bundle_align_mode 4
  .bundle_lock
  int3
  lock
  .bundle_unlock
# CHECK: [[#@LINE+1]]:3: error: instruction prefix cannot be the last instruction of a .bundle_lock group
  cmpxchgl %r12d, (%rbx)

#--- unsupported.s
# FORMAT: [[#@LINE+2]]:3: error: aligned bundling is not supported by this object file format
# TARGET: [[#@LINE+1]]:3: error: aligned bundling is not supported by this target
  .bundle_align_mode 4

#--- bundle-with-align-branch.s
## Instruction bundling cannot be combined with branch alignment.
# CHECK: [[#@LINE+1]]:3: error: .bundle_align_mode is incompatible with branch alignment
  .bundle_align_mode 5
  imull $17, %ebx, %ebp
