; Buffer intrinsics that take their resource as a <4 x i32> are deprecated.
; Check that codegen warns once per intrinsic and names the replacement.
;
; The --implicit-check-not=warning: ensures we don't have extra warnings.

; RUN: llc -mtriple=amdgpu9.00-amd-mesa3d -filetype=null < %s 2>&1 | \
; RUN:   FileCheck --implicit-check-not=warning: %s
; RUN: llc -mtriple=amdgpu9.00-amd-mesa3d -filetype=null \
; RUN:   -amdgpu-warn-deprecated-buffer-intrinsics=0 < %s 2>&1 | \
; RUN:   FileCheck -check-prefix=OFF --allow-empty --implicit-check-not=warning: %s

; CHECK: warning: 'llvm.amdgcn.raw.buffer.atomic.add' is deprecated and will be removed in a future LLVM release; use 'llvm.amdgcn.raw.ptr.buffer.atomic.add' with a 'ptr addrspace(8)' resource created by 'llvm.amdgcn.make.buffer.rsrc' instead
; CHECK: warning: 'llvm.amdgcn.raw.buffer.load' is deprecated {{.*}} use 'llvm.amdgcn.raw.ptr.buffer.load' {{.*}}
; CHECK: warning: 'llvm.amdgcn.raw.tbuffer.load' is deprecated {{.*}} use 'llvm.amdgcn.raw.ptr.tbuffer.load' {{.*}}
; CHECK: warning: 'llvm.amdgcn.s.buffer.load' is deprecated {{.*}} use 'llvm.amdgcn.ptr.s.buffer.load' {{.*}}
; CHECK: warning: 'llvm.amdgcn.struct.buffer.store' is deprecated {{.*}} use 'llvm.amdgcn.struct.ptr.buffer.store' {{.*}}

; OFF-NOT: {{.}}

; Two manglings of llvm.amdgcn.s.buffer.load, to check that the warning is
; emitted per intrinsic rather than per declaration.
define amdgpu_ps float @s_buffer_load_two_manglings(<4 x i32> inreg %rsrc) {
  %a = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 0, i32 0)
  %b = call <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32> %rsrc, i32 16, i32 0)
  %b0 = extractelement <4 x i32> %b, i32 0
  %sum = add i32 %a, %b0
  %r = bitcast i32 %sum to float
  ret float %r
}

define amdgpu_ps float @deprecated_buffer_ops(<4 x i32> inreg %rsrc, i32 %off) {
  %a = call float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32> %rsrc, i32 %off, i32 0, i32 0)
  %b = call float @llvm.amdgcn.raw.tbuffer.load.f32(<4 x i32> %rsrc, i32 %off, i32 0, i32 78, i32 0)
  %c = call i32 @llvm.amdgcn.raw.buffer.atomic.add.i32(i32 1, <4 x i32> %rsrc, i32 %off, i32 0, i32 0)
  %cf = bitcast i32 %c to float
  call void @llvm.amdgcn.struct.buffer.store.f32(float %a, <4 x i32> %rsrc, i32 0, i32 %off, i32 0, i32 0)
  %s0 = fadd float %a, %b
  %s1 = fadd float %s0, %cf
  ret float %s1
}

define amdgpu_ps float @ptr_buffer_load(ptr addrspace(8) inreg %rsrc, i32 %off) {
  %v = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %rsrc, i32 %off, i32 0, i32 0)
  ret float %v
}

; An image resource is a <8 x i32>, which must not be mistaken for a deprecated
; buffer resource.
define amdgpu_ps <4 x float> @image_load(<8 x i32> inreg %rsrc, i32 %s) {
  %v = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; Declared but never called.
declare void @llvm.amdgcn.struct.tbuffer.store.f32(float, <4 x i32>, i32, i32, i32, i32, i32)
