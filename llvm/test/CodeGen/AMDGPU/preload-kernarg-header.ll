; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx940 -amdgpu-kernarg-preload-count=1 -asm-verbose=0 < %s | FileCheck -check-prefixes=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx940 -amdgpu-kernarg-preload-count=1 -filetype=obj < %s | llvm-objdump --arch=amdgcn --mcpu=gfx940 --disassemble - | FileCheck -check-prefixes=GCN %s

; GCN: preload_kernarg_header
; GCN-COUNT-64: s_nop 0
define amdgpu_kernel void @preload_kernarg_header(ptr %arg) {
    store ptr %arg, ptr %arg
    ret void
}

; GCN: non_kernel_function
; GCN-NOT: s_nop 0
; GCN: flat_store
define void @non_kernel_function(ptr %arg) {
    store ptr %arg, ptr %arg
    ret void
}
