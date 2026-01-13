# RUN: not --crash llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s

# This test invokes .bundle_lock and then switches to a different section
# w/o the appropriate unlock.

# CHECK: ERROR: Unterminated .bundle_lock

  .bundle_align_mode 3
  .section text1, "x"
  imull $17, %ebx, %ebp
  .bundle_lock
  imull $17, %ebx, %ebp

  .section text2, "x"
  imull $17, %ebx, %ebp

