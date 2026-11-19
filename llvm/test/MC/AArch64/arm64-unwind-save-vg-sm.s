// RUN: llvm-mc -filetype=obj -triple aarch64-pc-windows-msvc %s -o - | \
// RUN: llvm-readobj --unwind - | FileCheck --strict-whitespace %s

// CHECK-LABEL: Function: save_vg_sm
// CHECK: Prologue [
// CHECK-NEXT: 0xe710c3            ; str vg_sm_val, [sp, #24]
// CHECK-NEXT: 0xe4                ; end
// CHECK-LABEL: Function: save_vg_sm_high_offset
// CHECK: Prologue [
// CHECK-NEXT: 0xe750c1            ; str vg_sm_val, [sp, #1032]
// CHECK-NEXT: 0xe4                ; end
// CHECK-LABEL: Function: save_vg_sm_max_offset
// CHECK: Prologue [
// CHECK-NEXT: 0xe770ff            ; str vg_sm_val, [sp, #2040]
// CHECK-NEXT: 0xe4                ; end

.text
.globl save_vg_sm
.seh_proc save_vg_sm
save_vg_sm:
  str x0, [sp, #24]
  .seh_save_vg_sm 24
  .seh_endprologue
  ret
  .seh_endproc

.globl save_vg_sm_high_offset
.seh_proc save_vg_sm_high_offset
save_vg_sm_high_offset:
  str x0, [sp, #1032]
  .seh_save_vg_sm 1032
  .seh_endprologue
  ret
  .seh_endproc

.globl save_vg_sm_max_offset
.seh_proc save_vg_sm_max_offset
save_vg_sm_max_offset:
  str x0, [sp, #2040]
  .seh_save_vg_sm 2040
  .seh_endprologue
  ret
  .seh_endproc
