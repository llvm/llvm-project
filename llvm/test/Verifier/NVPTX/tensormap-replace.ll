; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; The %ord operand of the indexed-field tensormap.replace intrinsics must be in
; the range [0, 5), per the PTX ISA (`tensormap.replace`: "The only valid values
; for `ord` are [0..4]").

declare void @llvm.nvvm.tensormap.replace.box.dim.p1(ptr addrspace(1) %addr, i32 immarg %ord, i32 %new_value)
declare void @llvm.nvvm.tensormap.replace.global.dim.p1(ptr addrspace(1) %addr, i32 immarg %ord, i32 %new_value)
declare void @llvm.nvvm.tensormap.replace.element.stride.p1(ptr addrspace(1) %addr, i32 immarg %ord, i32 %new_value)
declare void @llvm.nvvm.tensormap.replace.global.stride.p1(ptr addrspace(1) %addr, i32 immarg %ord, i64 %new_value)

define void @test_tensormap_replace_ord_out_of_range(ptr addrspace(1) %addr, i32 %v32, i64 %v64) {
  ; CHECK: immarg value 5 for arg 1 out of range [0,5)
  call void @llvm.nvvm.tensormap.replace.box.dim.p1(ptr addrspace(1) %addr, i32 5, i32 %v32)

  ; CHECK: immarg value 6 for arg 1 out of range [0,5)
  call void @llvm.nvvm.tensormap.replace.global.dim.p1(ptr addrspace(1) %addr, i32 6, i32 %v32)

  ; CHECK: immarg value 5 for arg 1 out of range [0,5)
  call void @llvm.nvvm.tensormap.replace.element.stride.p1(ptr addrspace(1) %addr, i32 5, i32 %v32)

  ; CHECK: immarg value 6 for arg 1 out of range [0,5)
  call void @llvm.nvvm.tensormap.replace.global.stride.p1(ptr addrspace(1) %addr, i32 6, i64 %v64)

  ret void
}
