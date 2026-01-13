# RUN: split-file %s %t
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/lock-without-mode.s     2>&1 | FileCheck %t/lock-without-mode.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/mode-without-arg.s      2>&1 | FileCheck %t/mode-without-arg.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/unlock-without-lock.s   2>&1 | FileCheck %t/unlock-without-lock.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/bad-lock-option.s       2>&1 | FileCheck %t/bad-lock-option.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/switch-section-locked.s 2>&1 | FileCheck %t/switch-section-locked.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/group-too-large.s       2>&1 | FileCheck %t/group-too-large.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 -mc-relax-all %t/group-too-large.s 2>&1 | FileCheck %t/group-too-large.s
# RUN: not llvm-mc -filetype=obj -triple x86_64 %t/nested-lock.s           2>&1 | FileCheck %t/nested-lock.s


## .bundle_lock can't come without a .bundle_align_mode before it
#--- lock-without-mode.s
  imull $17, %ebx, %ebp
# CHECK: [[#@LINE+1]]:3: error: .bundle_lock forbidden when bundling is disabled
  .bundle_lock

## .bundle_align_mode needs a following integer value
#--- mode-without-arg.s
# CHECK: [[#@LINE+1]]:21: error: unknown token in expression
  .bundle_align_mode
  imull $17, %ebx, %ebp

## .bundle_unlock can't come without a .bundle_lock before it
#--- unlock-without-lock.s
  .bundle_align_mode 3
  imull $17, %ebx, %ebp
# CHECK: [[#@LINE+1]]:3: error: .bundle_unlock without matching lock
  .bundle_unlock

## .bundle_lock can only take one `align_to_end` flag or no flag.
#--- bad-lock-option.s
  .bundle_align_mode 4
# CHECK: [[#@LINE+1]]:16: error: invalid option for `.bundle_lock`
  .bundle_lock 5
  imull $17, %ebx, %ebp
  .bundle_unlock

## This test invokes .bundle_lock and then switches to a different section
## w/o the appropriate unlock.
#--- switch-section-locked.s
  .bundle_align_mode 3
  .section text1, "x"
  imull $17, %ebx, %ebp
  .bundle_lock
  imull $17, %ebx, %ebp

# CHECK: [[#@LINE+1]]:3: error: unterminated .bundle_lock
  .section text2, "x"
  imull $17, %ebx, %ebp

## bundle lock size cannot be bigger than the align mode size
#--- group-too-large.s
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
# CHECK: [[#@LINE+1]]:3: error: fragment can't be larger than a bundle size
  .bundle_unlock

## test that nested lock is emitting the right error.
#--- nested-lock.s
  .bundle_align_mode 4
foo:
## bundle alignment mode can be set more than once.
  .bundle_align_mode 4
  .bundle_lock
# CHECK: [[#@LINE+1]]:3: error: nested .bundle_lock is not allowed
  .bundle_lock
  .bundle_unlock
  .bundle_unlock
