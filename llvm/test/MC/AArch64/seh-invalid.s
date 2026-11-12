// RUN: not llvm-mc -triple aarch64-pc-windows-msvc -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s

.text
.seh_proc invalid_alignment
  .seh_save_vg_sm 7
// CHECK: error: offset must be a multiple of 8
  .seh_endproc

.seh_proc invalid_range
  .seh_save_vg_sm 2048
// CHECK: error: offset must be between 0 and 2040 bytes
  .seh_endproc
