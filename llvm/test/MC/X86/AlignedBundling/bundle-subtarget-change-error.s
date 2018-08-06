# RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mcpu=pentiumpro %s -o - 2>&1 | FileCheck %s
# RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mcpu=pentiumpro -mc-relax-all %s -o - 2>&1 | FileCheck %s

# Switching mode will change subtarget, which we can't do within a bundle
  .text
  .code64
  .bundle_align_mode 4
foo:
  pushq   %rbp
  .bundle_lock
  addl    %ebp, %eax
  .code32
  movb  $0x0, (%si)
  .bundle_unlock

CHECK:  LLVM ERROR: A Bundle can only have one Subtarget.
