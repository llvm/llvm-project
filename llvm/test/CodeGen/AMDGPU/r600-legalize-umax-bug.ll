; RUN: llc -march=r600 -mcpu=cypress -start-after safe-stack %s -o - | FileCheck %s
; Don't crash

; CHECK: MAX_UINT
define amdgpu_kernel void @test(i64 addrspace(1)* %out) {
bb:
  store i64 2, i64 addrspace(1)* %out
  %tmp = load i64, i64 addrspace(1)* %out
  br label %jump

jump:                                             ; preds = %bb
  %tmp1 = icmp ugt i64 %tmp, 4
  %umax = select i1 %tmp1, i64 %tmp, i64 4
  store i64 %umax, i64 addrspace(1)* %out
  ret void
}
