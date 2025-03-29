; RUN: opt -passes=loop-idiom -S < %s -mtriple=r600 -mcpu=redwood | FileCheck --check-prefix=R600 --check-prefix=FUNC %s
; RUN: opt -passes=loop-idiom -S < %s -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs| FileCheck --check-prefix=SI --check-prefix=FUNC %s
; RUN: opt -passes=loop-idiom -S < %s -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs| FileCheck --check-prefix=SI --check-prefix=FUNC %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; Make sure loop-idiom doesn't create memcpy or memset.  There are no library
; implementations of these for R600.

; FUNC: @no_memcpy
; R600-NOT: {{^}}llvm.memcpy
; SI-NOT: {{^}}llvm.memcpy
define amdgpu_kernel void @no_memcpy(ptr addrspace(3) %in, i32 %size) {
entry:
  %dest = alloca i8, i32 32, addrspace(5)
  br label %for.body

for.body:
  %0 = phi i32 [0, %entry], [%4, %for.body]
  %1 = getelementptr i8, ptr addrspace(3) %in, i32 %0
  %2 = getelementptr i8, ptr addrspace(5) %dest, i32 %0
  %3 = load i8, ptr addrspace(3) %1
  store i8 %3, ptr addrspace(5) %2
  %4 = add i32 %0, 1
  %5 = icmp eq i32 %4, %size
  br i1 %5, label %for.end, label %for.body

for.end:
  ret void
}

; FUNC: @no_memset
; R600-NOT: {{^}}llvm.memset
; R600-NOT: {{^}}memset_pattern16:
; SI-NOT: {{^}}llvm.memset
; SI-NOT: {{^}}memset_pattern16:
define amdgpu_kernel void @no_memset(i32 %size) {
entry:
  %dest = alloca i8, i32 32, addrspace(5)
  br label %for.body

for.body:
  %0 = phi i32 [0, %entry], [%2, %for.body]
  %1 = getelementptr i8, ptr addrspace(5) %dest, i32 %0
  store i8 0, ptr addrspace(5) %1
  %2 = add i32 %0, 1
  %3 = icmp eq i32 %2, %size
  br i1 %3, label %for.end, label %for.body

for.end:
  ret void
}
