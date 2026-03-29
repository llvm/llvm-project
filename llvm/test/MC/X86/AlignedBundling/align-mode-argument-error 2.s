# RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s

# Missing .bundle_align_mode argument
# CHECK: error: unknown token

  .bundle_align_mode
  imull $17, %ebx, %ebp

