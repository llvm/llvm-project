; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

define void @tcgen05_mma_invalid_flag_values(ptr addrspace(6) %dtmem, ptr addrspace(6) %atensor, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d) {
  ; CHECK: immarg value 5 for arg 5 out of range [0,4)
  call void @llvm.nvvm.tcgen05.mma.shared(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, i32 5, i32 1, i32 3)
  ; CHECK: immarg value 0 for arg 6 out of range [1,3)
  call void @llvm.nvvm.tcgen05.mma.shared(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, i32 3, i32 0, i32 3)
  ; CHECK: immarg value 3 for arg 6 out of range [1,3)
  call void @llvm.nvvm.tcgen05.mma.shared(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, i32 3, i32 3, i32 3)
  ; CHECK: immarg value 5 for arg 7 out of range [0,4)
  call void @llvm.nvvm.tcgen05.mma.shared(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, i32 3, i32 2, i32 5)
  ; CHECK: immarg value 2 for arg 7 out of range [0,2)
  call void @llvm.nvvm.tcgen05.mma.tensor.ashift(ptr addrspace(6) %dtmem, ptr addrspace(6) %atensor, i64 %b, i32 %idesc, i1 %enable_inp_d, i32 3, i32 2, i32 2)
  ret void
}

define void @tcgen05_mma_disable_output_lane_invalid_flag_values(ptr addrspace(6) %dtmem, ptr addrspace(6) %atensor, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, <4 x i32> %disable_output_lanev4) {
  ; CHECK: immarg value 5 for arg 6 out of range [0,4)
  call void @llvm.nvvm.tcgen05.mma.tensor.disable_output_lane.cg1(ptr addrspace(6) %dtmem, ptr addrspace(6) %atensor, i64 %b, i32 %idesc, i1 %enable_inp_d, <4 x i32> %disable_output_lanev4, i32 5, i32 0)
  ; CHECK: immarg value 5 for arg 7 out of range [0,4)
  call void @llvm.nvvm.tcgen05.mma.tensor.disable_output_lane.cg1(ptr addrspace(6) %dtmem, ptr addrspace(6) %atensor, i64 %b, i32 %idesc, i1 %enable_inp_d, <4 x i32> %disable_output_lanev4, i32 0, i32 5)
  ret void
}

define void @tcgen05_mma_ws_invalid_flag_values(ptr addrspace(6) %dtmem, ptr addrspace(6) %atensor, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d) {
  ; CHECK: immarg value 5 for arg 5 out of range [0,4)
  call void @llvm.nvvm.tcgen05.mma.ws.shared(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, i32 5, i32 0, i32 0)
  ; CHECK: immarg value 5 for arg 6 out of range [0,4)
  call void @llvm.nvvm.tcgen05.mma.ws.shared(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, i32 0, i32 5, i32 0)
  ; CHECK: immarg value 5 for arg 7 out of range [0,4)
  call void @llvm.nvvm.tcgen05.mma.ws.shared(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, i32 0, i32 0, i32 5)
  ret void
}
