; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=tahiti < %s
; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=tonga < %s

; This works because promote allocas pass replaces these with LDS atomics.

; Private atomics have no real use, but at least shouldn't crash on it.
define amdgpu_kernel void @atomicrmw_private(i32 addrspace(1)* %out, i32 %in) nounwind {
entry:
  %tmp = alloca [2 x i32]
  %tmp1 = getelementptr inbounds [2 x i32], [2 x i32]* %tmp, i32 0, i32 0
  %tmp2 = getelementptr inbounds [2 x i32], [2 x i32]* %tmp, i32 0, i32 1
  store i32 0, i32* %tmp1
  store i32 1, i32* %tmp2
  %tmp3 = getelementptr inbounds [2 x i32], [2 x i32]* %tmp, i32 0, i32 %in
  %tmp4 = atomicrmw add i32* %tmp3, i32 7 acq_rel
  store i32 %tmp4, i32 addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @cmpxchg_private(i32 addrspace(1)* %out, i32 %in) nounwind {
entry:
  %tmp = alloca [2 x i32]
  %tmp1 = getelementptr inbounds [2 x i32], [2 x i32]* %tmp, i32 0, i32 0
  %tmp2 = getelementptr inbounds [2 x i32], [2 x i32]* %tmp, i32 0, i32 1
  store i32 0, i32* %tmp1
  store i32 1, i32* %tmp2
  %tmp3 = getelementptr inbounds [2 x i32], [2 x i32]* %tmp, i32 0, i32 %in
  %tmp4 = cmpxchg i32* %tmp3, i32 0, i32 1 acq_rel monotonic
  %val = extractvalue { i32, i1 } %tmp4, 0
  store i32 %val, i32 addrspace(1)* %out
  ret void
}
