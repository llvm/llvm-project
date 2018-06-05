# RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o /dev/null 2>&1 | FileCheck %s

  .cg_profile a, .L.temp, 32

# CHECK:      cgprofile-error.s:3:18: error: Reference to undefined temporary symbol `.L.temp`
# CHECK-NEXT:   .cg_profile a, .L.temp, 32
# CHECK-NEXT:                  ^
