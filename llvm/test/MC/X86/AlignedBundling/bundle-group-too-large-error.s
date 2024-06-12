# RUN: not --crash llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mc-relax-all %s -o /dev/null

# CHECK: ERROR: Fragment can't be larger than a bundle size

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
  .bundle_unlock

