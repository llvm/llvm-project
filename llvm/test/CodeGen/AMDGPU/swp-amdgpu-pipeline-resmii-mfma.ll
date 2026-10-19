; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -amdgpu-enable-pipeliner -debug-only=pipeliner %s -filetype=null 2>&1 | FileCheck %s
; REQUIRES: asserts

; Resource-bound II set by a single functional unit, the MFMA (XDL) matrix pipe.
; Two independent accumulators each issue one MFMA per iteration; each MFMA holds
; the XDL pipe for 16 cycles, so res=32 on that one unit. That exceeds both the
; 3-micro-op issue bound and the single-MFMA recurrence (rec=17), so the II is
; resource-bound on the XDL pipe rather than on issue width or recurrence.
; CHECK: MII = 32 MAX_II = 42 (rec=17, res=32)

define amdgpu_kernel void @swp_amdgpu_pipeline_resmii_mfma(i32 %arg, ptr addrspace(3) %p) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %phiA = phi <16 x float> [ zeroinitializer, %bb ], [ %mfmaA, %bb1 ]
  %phiB = phi <16 x float> [ zeroinitializer, %bb ], [ %mfmaB, %bb1 ]
  %iv = phi i32 [ 0, %bb ], [ %arg, %bb1 ]
  %load = load <4 x float>, ptr addrspace(3) %p, align 16
  %elt = extractelement <4 x float> %load, i64 0
  %mfmaA = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float %elt, float 0.000000e+00, <16 x float> %phiA, i32 0, i32 0, i32 0)
  %mfmaB = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float %elt, float 0.000000e+00, <16 x float> %phiB, i32 0, i32 0, i32 0)
  %icmp = icmp eq i32 %iv, 0
  br i1 %icmp, label %exit, label %bb1

exit:                                             ; preds = %bb1
  %lcssaA = phi <16 x float> [ %mfmaA, %bb1 ]
  %lcssaB = phi <16 x float> [ %mfmaB, %bb1 ]
  store <16 x float> %lcssaA, ptr addrspace(3) %p, align 64
  store <16 x float> %lcssaB, ptr addrspace(3) %p, align 64
  ret void
}
