# RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s

# CHECK: error: nested .bundle_lock is not allowed

# Will be bundle-aligning to 16 byte boundaries
  .bundle_align_mode 4
foo:
# Test that bundle alignment mode can be set more than once.
  .bundle_align_mode 4
  .bundle_lock
  .bundle_lock
  .bundle_unlock
  .bundle_unlock
