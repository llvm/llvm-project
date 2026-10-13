// RUN: not llvm-mc -triple aarch64-pc-windows-msvc -filetype=obj %s -o %t.obj 2>&1 | FileCheck %s
// CHECK-COUNT-2: .seh_end_c must appear once in a prologue

.text
// A prologue must not contain more than one end_c marker.
duplicate:
  .seh_proc duplicate
  .seh_end_c
  .seh_end_c
  .seh_endprologue
  nop
  .seh_endproc

// end_c describes a phantom prologue and must be rejected inside an epilogue.
epilogue:
  .seh_proc epilogue
  sub sp, sp, #16
  .seh_stackalloc 16
  .seh_endprologue
  .seh_startepilogue
  add sp, sp, #16
  .seh_stackalloc 16
  .seh_end_c
  .seh_endepilogue
  ret
  .seh_endproc