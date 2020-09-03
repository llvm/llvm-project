# RUN: not --crash llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s

# .bundle_lock can't come without a .bundle_align_mode before it

# CHECK: ERROR: .bundle_lock forbidden when bundling is disabled

  imull $17, %ebx, %ebp
  .bundle_lock


