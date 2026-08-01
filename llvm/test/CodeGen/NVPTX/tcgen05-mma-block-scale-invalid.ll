; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

define void @tcgen05_mma_block_scale_invalid_flags(ptr addrspace(6) %dtmem, ptr addrspace(6) %atensor, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, ptr addrspace(6) %scale_a, ptr addrspace(6) %scale_b) {
  ; CHECK: immarg value 0 for arg 7 out of range [1,3)
  call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale.block32(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, ptr addrspace(6) %scale_a, ptr addrspace(6) %scale_b, i32 0, i32 0)
  ; CHECK: immarg value 5 for arg 8 out of range [0,4)
  call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale.block32(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, ptr addrspace(6) %scale_a, ptr addrspace(6) %scale_b, i32 1, i32 5)
  ; CHECK: immarg value 0 for arg 7 out of range [1,3)
  call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale.block32(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, ptr addrspace(6) %scale_a, ptr addrspace(6) %scale_b, i32 0, i32 0)
  ; CHECK: immarg value 5 for arg 8 out of range [0,4)
  call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale.block32(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, ptr addrspace(6) %scale_a, ptr addrspace(6) %scale_b, i32 1, i32 5)
  ; CHECK: immarg value 0 for arg 7 out of range [1,3)
  call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block32(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, ptr addrspace(6) %scale_a, ptr addrspace(6) %scale_b, i32 0, i32 0)
  ; CHECK: immarg value 5 for arg 8 out of range [0,4)
  call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block32(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, ptr addrspace(6) %scale_a, ptr addrspace(6) %scale_b, i32 1, i32 5)
  call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block16(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, ptr addrspace(6) %scale_a, ptr addrspace(6) %scale_b, i32 0, i32 0)
  ; CHECK: immarg value 5 for arg 8 out of range [0,4)
  call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block16(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, ptr addrspace(6) %scale_a, ptr addrspace(6) %scale_b, i32 1, i32 5)
  ret void
}
