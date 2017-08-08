; RUN: opt -basicaa -loop-idiom -S < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=R600 --check-prefix=FUNC %s
; RUN: opt -basicaa -loop-idiom -S < %s -march=amdgcn -mcpu=SI -verify-machineinstrs| FileCheck --check-prefix=SI --check-prefix=FUNC %s
; RUN: opt -basicaa -loop-idiom -S < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs| FileCheck --check-prefix=SI --check-prefix=FUNC %s


; Make sure loop-idiom doesn't create memcpy or memset.  There are no library
; implementations of these for R600.

; FUNC: @no_memcpy
; R600-NOT: {{^}}llvm.memcpy
; SI-NOT: {{^}}llvm.memcpy
define amdgpu_kernel void @no_memcpy(i8 addrspace(3)* %in, i32 %size) {
entry:
  %dest = alloca i8, i32 32
  br label %for.body

for.body:
  %0 = phi i32 [0, %entry], [%4, %for.body]
  %1 = getelementptr i8, i8 addrspace(3)* %in, i32 %0
  %2 = getelementptr i8, i8* %dest, i32 %0
  %3 = load i8, i8 addrspace(3)* %1
  store i8 %3, i8* %2
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
  %dest = alloca i8, i32 32
  br label %for.body

for.body:
  %0 = phi i32 [0, %entry], [%2, %for.body]
  %1 = getelementptr i8, i8* %dest, i32 %0
  store i8 0, i8* %1
  %2 = add i32 %0, 1
  %3 = icmp eq i32 %2, %size
  br i1 %3, label %for.end, label %for.body

for.end:
  ret void
}
