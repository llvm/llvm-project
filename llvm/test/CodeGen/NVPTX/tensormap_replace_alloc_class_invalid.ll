; RUN: split-file %s %t
; RUN: not --crash llc -mtriple=nvptx64 -mcpu=sm_90a -mattr=+ptx94 -filetype=null %t/alloc-large.ll 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple=nvptx64 -mcpu=sm_100f -mattr=+ptx88 -filetype=null %t/alloc-same.ll 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tensormap.replace.global.address

;--- alloc-large.ll
define void @tensormap_replace_global_address_alloc_large(ptr addrspace(1) %addr, i64 %value) {
  call void @llvm.nvvm.tensormap.replace.global.address.p1(ptr addrspace(1) %addr, i64 %value, i32 1)
  ret void
}

;--- alloc-same.ll
define void @tensormap_replace_global_address_alloc_same(ptr addrspace(3) %addr, i64 %value) {
  call void @llvm.nvvm.tensormap.replace.global.address.p3(ptr addrspace(3) %addr, i64 %value, i32 2)
  ret void
}
