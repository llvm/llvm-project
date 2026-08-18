; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -pass-remarks-missed=si-pin-regs \
; RUN:   < %s 2>&1 | FileCheck %s

; A pin is a hint, so a value the allocator places elsewhere is not reported --
; there is nothing the user could do about it. A pin that cannot be turned into
; a hint at all is reported, since nothing in the source says why it was
; dropped.

declare <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32>, i32 immarg)
declare <4 x i32> @llvm.amdgcn.pin.vgpr.v4i32(<4 x i32>, i32 immarg)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16(i1, <16 x bfloat>, i1, <16 x bfloat>, i16, <8 x float>, i1, i1)

; A loop-carried accumulator reaches its pin through a PHI. A hint covers the
; whole chain, so this is silent.
; CHECK-NOT: remark:
define amdgpu_kernel void @pin_through_phi(ptr addrspace(1) %o, ptr addrspace(1) %p) {
entry:
  %z = load <8 x i32>, ptr addrspace(1) %p, align 32
  %i = call <8 x i32> @llvm.amdgcn.pin.vgpr.v8i32(<8 x i32> %z, i32 108)
  br label %loop

loop:
  %a = phi <8 x i32> [ %i, %entry ], [ %d.i, %loop ]
  %c = bitcast <8 x i32> %a to <8 x float>
  %d = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16(i1 false, <16 x bfloat> zeroinitializer, i1 false, <16 x bfloat> zeroinitializer, i16 0, <8 x float> %c, i1 false, i1 false)
  %d.i = bitcast <8 x float> %d to <8 x i32>
  br i1 false, label %exit, label %loop

exit:
  store <8 x i32> %a, ptr addrspace(1) %o
  ret void
}

; An odd start for a 4-VGPR value names no aligned tuple, so there is nothing to
; hint at and the pin is dropped.
; CHECK: remark: {{.*}} pin to v101 was dropped: no register tuple of this width and alignment starts there
; CHECK-NOT: remark:
define amdgpu_kernel void @pin_to_unaligned(ptr addrspace(1) %o, <4 x i32> %v) {
  %p = call <4 x i32> @llvm.amdgcn.pin.vgpr.v4i32(<4 x i32> %v, i32 101)
  store <4 x i32> %p, ptr addrspace(1) %o
  ret void
}
