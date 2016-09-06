; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -relocation-model=static < %s | FileCheck %s

@lds = external addrspace(3) global [4 x i32]

; Function Attrs: nounwind

; Offset folding is an optimization done for global variables with relocations,
; which allows you to store the offset in the r_addend of the relocation entry.
; The offset is apllied to the variables address at link time, which eliminates
; the need to emit shader instructions to do this calculation.
; We don't use relocations for local memory, so we should never fold offsets
; for local memory globals.

; CHECK-LABEL: lds_no_offset:
; CHECK ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:4
define void @lds_no_offset() {
entry:
  %ptr = getelementptr [4 x i32], [4 x i32] addrspace(3)* @lds, i32 0, i32 1
  store i32 0, i32 addrspace(3)* %ptr
  ret void
}
