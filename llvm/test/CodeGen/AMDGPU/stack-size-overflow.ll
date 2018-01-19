; RUN: not llc -march=amdgcn < %s 2>&1 | FileCheck -check-prefix=ERROR %s
; RUN: not llc -march=amdgcn < %s | FileCheck -check-prefix=GCN %s

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) #1

; ERROR: error: stack size limit exceeded (4294967296) in stack_size_limit
; GCN: ; ScratchSize: 4294967296
define amdgpu_kernel void @stack_size_limit() #0 {
entry:
  %alloca = alloca [1073741823 x i32], align 4
  %bc = bitcast [1073741823 x i32]* %alloca to i8*
  call void @llvm.memset.p0i8.i32(i8* %bc, i8 9, i32 1073741823, i1 true)
  ret void
}
