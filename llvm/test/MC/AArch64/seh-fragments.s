// RUN: llvm-mc -triple aarch64-pc-windows-msvc -filetype=obj %s -o %t.obj
// RUN: llvm-readobj --unwind %t.obj | FileCheck %s

// A body-only fragment must encode end_c followed by the parent unwind recipe,
// even though no stack adjustment or register save appears in its code.
// CHECK: Function: body_only
// CHECK: Prologue [
// CHECK-NEXT: 0xe5                ; end_c
// CHECK-NEXT: 0xd60c              ; stp x19, lr, [sp, #96]
// CHECK-NEXT: 0x07                ; sub sp, #112
// CHECK-NEXT: 0xe4                ; end

.text
.globl body_only
body_only:
  .seh_proc body_only
  .seh_stackalloc 112
  .seh_save_lrpair x19, 96
  .seh_end_c
  .seh_endprologue
  nop
  nop
  .seh_endproc

// A phantom prologue must coexist with a real epilogue whose unwind operations
// still describe the physical restores and stack adjustment.
// CHECK: Function: with_epilogue
// CHECK: Prologue [
// CHECK-NEXT: 0xe5                ; end_c
// CHECK-NEXT: 0xd60c              ; stp x19, lr, [sp, #96]
// CHECK-NEXT: 0x07                ; sub sp, #112
// CHECK-NEXT: 0xe4                ; end
// CHECK: Epilogue [
// CHECK-NEXT: 0xd60c              ; ldp x19, lr, [sp, #96]
// CHECK-NEXT: 0x07                ; add sp, #112
// CHECK-NEXT: 0xe4                ; end

.globl with_epilogue
with_epilogue:
  .seh_proc with_epilogue
  .seh_stackalloc 112
  .seh_save_lrpair x19, 96
  .seh_end_c
  .seh_endprologue
  nop
  .seh_startepilogue
  ldp x19, lr, [sp, #96]
  .seh_save_lrpair x19, 96
  add sp, sp, #112
  .seh_stackalloc 112
  .seh_endepilogue
  ret
  .seh_endproc

// Large-function segmentation must preserve the parent recipe in each segment
// without duplicating the explicit end_c marker.
// CHECK: Function: large_fragment
// CHECK: FunctionLength: 1048572
// CHECK: Prologue [
// CHECK-NEXT: 0xe5                ; end_c
// CHECK-NEXT: 0xd60c              ; stp x19, lr, [sp, #96]
// CHECK: Function: {{.*}} (0x100014)
// CHECK: Prologue [
// CHECK-NEXT: 0xe5                ; end_c
// CHECK-NEXT: 0xd60c              ; stp x19, lr, [sp, #96]
// CHECK-NEXT: 0x07                ; sub sp, #112
// CHECK-NEXT: 0xe4                ; end

.globl large_fragment
large_fragment:
  .seh_proc large_fragment
  .seh_stackalloc 112
  .seh_save_lrpair x19, 96
  .seh_end_c
  .seh_endprologue
  .rept 262200
  nop
  .endr
  .seh_endproc
