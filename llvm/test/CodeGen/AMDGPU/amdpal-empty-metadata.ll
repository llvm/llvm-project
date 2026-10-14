; RUN: llc -mtriple=amdgpu9.00-amd-amdpal -filetype=obj < %s | llvm-readobj --notes - | FileCheck %s

; A regular function produces no PAL pipeline metadata. Do not try to serialize
; the empty MsgPack document when emitting an ELF object.
; CHECK: Format: elf64-amdgpu
; CHECK-NOT: METADATA
; CHECK: Type: NT_AMD_HSA_ISA_NAME
; CHECK-NOT: METADATA

define i32 @f(i32 %x) {
  %y = add i32 %x, 1
  ret i32 %y
}
