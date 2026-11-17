; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define void @test_tensormap_alloc_class_out_of_range(ptr addrspace(1) %addr, i64 %value) {
  ; CHECK: immarg value 3 for arg 2 out of range [0,3)
  call void @llvm.nvvm.tensormap.replace.global.address.p1(ptr addrspace(1) %addr, i64 %value, i32 3)

  ; CHECK: immarg value 2 for arg 1 out of range [0,2)
  call void @llvm.nvvm.tensormap.replace.alloc.class.p1(ptr addrspace(1) %addr, i32 2)

  ret void
}
