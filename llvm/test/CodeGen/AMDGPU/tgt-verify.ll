; RUN: not llvm-tgt-verify %s -mtriple=amdgcn |& FileCheck %s

define amdgpu_kernel void @test_mfma_f32_32x32x1f32_vecarg(ptr addrspace(1) %arg) #0 {
; CHECK: Not uniform: %in.f32 = load <32 x float>, ptr addrspace(1) %gep, align 128
; CHECK-NEXT: MFMA in control flow
; CHECK-NEXT:   %mfma = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.000000e+00, float 2.000000e+00, <32 x float> %in.f32, i32 1, i32 2, i32 3)
s:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, ptr addrspace(1) %arg, i32 %tid
  %in.i32 = load <32 x i32>, ptr addrspace(1) %gep
  %in.f32 = load <32 x float>, ptr addrspace(1) %gep

  %0 = icmp eq <32 x i32> %in.i32, zeroinitializer
  %div.br = extractelement <32 x i1> %0, i32 0
  br i1 %div.br, label %if.3, label %else.0

if.3:
  br label %join

else.0:
  %mfma = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.000000e+00, float 2.000000e+00, <32 x float> %in.f32, i32 1, i32 2, i32 3)
  br label %join

join:
  ret void
}

define amdgpu_cs i32 @shader() {
; CHECK: Shaders must return void
  ret i32 0
}

define amdgpu_kernel void @store_const(ptr addrspace(4) %out, i32 %a, i32 %b) {
; CHECK: Undefined behavior: Write to memory in const addrspace
; CHECK-NEXT:   store i32 %r, ptr addrspace(4) %out, align 4
; CHECK-NEXT: Write to const memory
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
