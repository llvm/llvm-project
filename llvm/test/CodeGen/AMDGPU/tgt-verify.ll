; RUN: not llvm-tgt-verify %s -mtriple=amdgcn |& FileCheck %s

define amdgpu_cs i32 @shader() {
; CHECK: Shaders must return void
  ret i32 0
}

define amdgpu_kernel void @store_const(ptr addrspace(4) %out, i32 %a, i32 %b) {
; CHECK: Undefined behavior: Write to memory in const addrspace
; CHECK-NEXT:   store i32 %r, ptr addrspace(4) %out, align 4
  %r = add i32 %a, %b
  store i32 %r, ptr addrspace(4) %out
  ret void
}

define amdgpu_kernel void @kernel_callee(ptr %x) {
  ret void
}

define amdgpu_kernel void @kernel_caller(ptr %x) {
; CHECK: A kernel may not call a kernel
; CHECK-NEXT: ptr @kernel_caller
  call amdgpu_kernel void @kernel_callee(ptr %x)
  ret void
}


; Function Attrs: nounwind
define i65 @invalid_type(i65 %x) #0 {
; CHECK: Int type is invalid.
; CHECK-NEXT: %tmp2 = ashr i65 %x, 64
entry:
  %tmp2 = ashr i65 %x, 64
  ret i65 %tmp2
}

declare void @llvm.amdgcn.cs.chain.v3i32(ptr, i32, <3 x i32>, <3 x i32>, i32, ...)
declare amdgpu_cs_chain void @chain_callee(<3 x i32> inreg, <3 x i32>)

define amdgpu_cs void @no_unreachable(<3 x i32> inreg %a, <3 x i32> %b) {
; CHECK: llvm.amdgcn.cs.chain must be followed by unreachable
; CHECK-NEXT: call void (ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.p0.i32.v3i32.v3i32(ptr @chain_callee, i32 -1, <3 x i32> inreg %a, <3 x i32> %b, i32 0)
  call void(ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.v3i32(ptr @chain_callee, i32 -1, <3 x i32> inreg %a, <3 x i32> %b, i32 0)
  ret void
}
