; RUN: not llc -march=amdgcn < %s 2>&1 | FileCheck -check-prefix=ERROR %s
; RUN: not llc -march=amdgcn < %s | FileCheck -check-prefix=GCN %s

declare void @llvm.memset.p5.i32(ptr addrspace(5) nocapture, i8, i32, i32, i1) #1

; ERROR: error: <unknown>:0:0: stack frame size (131061) exceeds limit (131056) in function 'stack_size_limit_wave64'
; GCN: ; ScratchSize: 131061
define amdgpu_kernel void @stack_size_limit_wave64() #0 {
entry:
  %alloca = alloca [131057 x i8], align 1, addrspace(5)
  call void @llvm.memset.p5.i32(ptr addrspace(5) %alloca, i8 9, i32 131057, i32 1, i1 true)
  ret void
}

; ERROR: error: <unknown>:0:0: stack frame size (262117) exceeds limit (262112) in function 'stack_size_limit_wave32'
; GCN: ; ScratchSize: 262117
define amdgpu_kernel void @stack_size_limit_wave32() #1 {
entry:
  %alloca = alloca [262113 x i8], align 1, addrspace(5)
  call void @llvm.memset.p5.i32(ptr addrspace(5) %alloca, i8 9, i32 262113, i32 1, i1 true)
  ret void
}

; ERROR-NOT: error:
; GCN: ; ScratchSize: 131056
define amdgpu_kernel void @max_stack_size_wave64() #0 {
entry:
  %alloca = alloca [131052 x i8], align 1, addrspace(5)
  call void @llvm.memset.p5.i32(ptr addrspace(5) %alloca, i8 9, i32 131052, i32 1, i1 true)
  ret void
}

; ERROR-NOT: error:
; GCN: ; ScratchSize: 262112
define amdgpu_kernel void @max_stack_size_wave32() #1 {
entry:
  %alloca = alloca [262108 x i8], align 1, addrspace(5)
  call void @llvm.memset.p5.i32(ptr addrspace(5) %alloca, i8 9, i32 262108, i32 1, i1 true)
  ret void
}

attributes #0 = { "target-cpu" = "gfx900" }
attributes #1 = { "target-cpu" = "gfx1010" }
