# RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s

# Missing .bundle_align_mode argument
# CHECK: error: invalid option

  .bundle_align_mode 4
  .bundle_lock 5
  imull $17, %ebx, %ebp
  .bundle_unlock


