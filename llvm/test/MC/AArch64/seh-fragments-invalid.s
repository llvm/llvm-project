// RUN: not llvm-mc -triple aarch64-pc-windows-msvc -filetype=obj %s -o %t.obj 2>&1 | FileCheck %s
// CHECK-COUNT-3: .seh_end_c must appear once in a prologue or epilogue

.text
// A prologue must not contain more than one end_c marker.
duplicate:
  .seh_proc duplicate
  .seh_end_c
  .seh_end_c
  .seh_endprologue
  nop
  .seh_endproc

epilogue:
  .seh_proc epilogue
  sub sp, sp, #16
  .seh_stackalloc 16
  .seh_endprologue
  .seh_startepilogue
  add sp, sp, #16
  .seh_stackalloc 16
  .seh_end_c
  .seh_end_c
  .seh_endepilogue
  ret
  .seh_endproc

body:
  .seh_proc body
  sub sp, sp, #16
  .seh_stackalloc 16
  .seh_endprologue
  .seh_end_c
  nop
  .seh_endproc

// CHECK: error: Incorrect size for prologue_size prologue: 8 bytes of instructions in range, but .seh directives corresponding to 4 bytes
prologue_size:
  .seh_proc prologue_size
  sub sp, sp, #16
  .seh_stackalloc 16
  nop
  .seh_endprologue
  nop
  .seh_endproc

// CHECK: error: Incorrect size for epilogue_size epilogue: 8 bytes of instructions in range, but .seh directives corresponding to 4 bytes
epilogue_size:
  .seh_proc epilogue_size
  sub sp, sp, #16
  .seh_stackalloc 16
  .seh_endprologue
  .seh_startepilogue
  add sp, sp, #16
  .seh_stackalloc 16
  nop
  .seh_endepilogue
  ret
  .seh_endproc

// CHECK: error: Incorrect size for chained_prologue_size prologue: 4 bytes of instructions in range, but .seh directives corresponding to 8 bytes
chained_prologue_size:
  .seh_proc chained_prologue_size
  .seh_stackalloc 112
  .seh_save_lrpair x19, 96
  .seh_end_c
  stp x21, x22, [sp, #16]
  .seh_save_regp x21, 16
  .seh_nop
  .seh_endprologue
  nop
  .seh_endproc

// CHECK: error: Incorrect size for chained_epilogue_size epilogue: 4 bytes of instructions in range, but .seh directives corresponding to 8 bytes
chained_epilogue_size:
  .seh_proc chained_epilogue_size
  .seh_stackalloc 112
  .seh_save_lrpair x19, 96
  .seh_end_c
  stp x21, x22, [sp, #16]
  .seh_save_regp x21, 16
  .seh_endprologue
  nop
  .seh_startepilogue
  ldp x21, x22, [sp, #16]
  .seh_save_regp x21, 16
  .seh_nop
  .seh_end_c
  .seh_save_lrpair x19, 96
  .seh_stackalloc 112
  .seh_endepilogue
  b parent_epilogue
  .seh_endproc