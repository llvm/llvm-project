; XFAIL: *
; REQUIRES: asserts
; RUN: llc -march=amdgcn -mcpu=SI < %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s

define void @large_alloca(i32 addrspace(1)* %out, i32 %x, i32 %y) nounwind {
  %large = alloca [8192 x i32], align 4
  %gep = getelementptr [8192 x i32]* %large, i32 0, i32 8191
  store i32 %x, i32* %gep
  %gep1 = getelementptr [8192 x i32]* %large, i32 0, i32 %y
  %0 = load i32* %gep1
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

