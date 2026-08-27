; RUN: not --crash llc -o - -mcpu=sm_107f -mattr=+ptx94 -march=nvptx64 %s 2>&1 | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

define void @tcgen05_mma_i8_collector_b_sm107f(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d) {
; CHECK: LLVM ERROR: Cannot select:
  ; kind=i8(3), cta_group=1, collector_a=discard(0), collector_b=lastuse(1)
  call void @llvm.nvvm.tcgen05.mma.shared(ptr addrspace(6) %dtmem, i64 %ashared, i64 %b, i32 %idesc, i1 %enable_inp_d, i32 3, i32 1, i32 0, i32 1)
  ret void
}
