// RUN: llvm-mc -filetype=obj -triple amdgcn-- -mcpu=kaveri -show-encoding %s | llvm-readobj -r - | FileCheck %s

// CHECK: Relocations [
// CHECK: .rel.eh_frame {
// CHECK: R_AMDGPU_REL32 .text
// CHECK: }
// CHECK: .rel.debug_frame {
// CHECK: R_AMDGPU_ABS64 .text
// CHECK: }
// CHECK: ]

kernel:
  .cfi_startproc
  .cfi_sections .debug_frame, .eh_frame
  s_endpgm
  .cfi_endproc