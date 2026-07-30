; RUN: not llc -mtriple=amdgpu9.00-- < %s 2> %t-wave64.err | FileCheck -implicit-check-not=error -check-prefix=GCN %s
; RUN: not llc -mtriple=amdgpu10.10-- < %s 2> %t-wave32.err | FileCheck -implicit-check-not=error -check-prefix=GCN %s
; RUN: FileCheck -check-prefix=WAVE32-ERR %s < %t-wave32.err
; RUN: FileCheck -check-prefix=WAVE64-ERR %s < %t-wave64.err

declare void @llvm.memset.p5.i32(ptr addrspace(5) nocapture, i8, i32, i32, i1) #1

; WAVE64-ERR: error: <unknown>:0:0: stack frame size (131064) exceeds limit (131056) in function 'stack_size_limit_wave64'
; GCN: ; ScratchSize: 131064
define amdgpu_kernel void @stack_size_limit_wave64() {
entry:
  %alloca = alloca [131057 x i8], align 1, addrspace(5)
  call void @llvm.memset.p5.i32(ptr addrspace(5) %alloca, i8 9, i32 131057, i32 1, i1 true)
  ret void
}

; WAVE32-ERR: error: <unknown>:0:0: stack frame size (262120) exceeds limit (262112) in function 'stack_size_limit_wave32'
; GCN: ; ScratchSize: 262120
define amdgpu_kernel void @stack_size_limit_wave32() {
entry:
  %alloca = alloca [262113 x i8], align 1, addrspace(5)
  call void @llvm.memset.p5.i32(ptr addrspace(5) %alloca, i8 9, i32 262113, i32 1, i1 true)
  ret void
}

; GCN: ; ScratchSize: 131056
define amdgpu_kernel void @max_stack_size_wave64() {
entry:
  %alloca = alloca [131052 x i8], align 1, addrspace(5)
  call void @llvm.memset.p5.i32(ptr addrspace(5) %alloca, i8 9, i32 131052, i32 1, i1 true)
  ret void
}

; GCN: ; ScratchSize: 262112
define amdgpu_kernel void @max_stack_size_wave32() {
entry:
  %alloca = alloca [262108 x i8], align 1, addrspace(5)
  call void @llvm.memset.p5.i32(ptr addrspace(5) %alloca, i8 9, i32 262108, i32 1, i1 true)
  ret void
}
