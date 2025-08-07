# RUN: not --crash llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s

# .bundle_unlock can't come without a .bundle_lock before it

# CHECK: ERROR: .bundle_unlock without matching lock

  .bundle_align_mode 3
  imull $17, %ebx, %ebp
  .bundle_unlock


