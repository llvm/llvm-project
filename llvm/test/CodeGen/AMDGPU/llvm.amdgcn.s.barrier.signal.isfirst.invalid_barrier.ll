; RUN: not llc -mtriple=amdgpu12.50 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s

; ERR: error: <unknown>:0:0: in function invalid_barrier_id i1 (): s_barrier_signal_isfirst does not support user_cluster_barrier_id (-3)

define i1 @invalid_barrier_id() {
  %r = call i1 @llvm.amdgcn.s.barrier.signal.isfirst(i32 -3)
  ret i1 %r
}

