; RUN: not llc -verify-machineinstrs -march=amdgcn %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -march=amdgcn -mcpu=tonga %s -o /dev/null 2>&1 | FileCheck %s

; Make sure promote alloca pass doesn't crash

; CHECK: unsupported call

declare i32 @foo(i32*) nounwind

define amdgpu_kernel void @call_private(i32 addrspace(1)* %out, i32 %in) nounwind {
entry:
  %tmp = alloca [2 x i32]
  %tmp1 = getelementptr [2 x i32], [2 x i32]* %tmp, i32 0, i32 0
  %tmp2 = getelementptr [2 x i32], [2 x i32]* %tmp, i32 0, i32 1
  store i32 0, i32* %tmp1
  store i32 1, i32* %tmp2
  %tmp3 = getelementptr [2 x i32], [2 x i32]* %tmp, i32 0, i32 %in
  %val = call i32 @foo(i32* %tmp3) nounwind
  store i32 %val, i32 addrspace(1)* %out
  ret void
}
