; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -amdgpu-enable-pipeliner -pass-remarks-analysis=pipeliner %s -filetype=null 2>&1 | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -amdgpu-enable-pipeliner -pipeliner-max-mii=64 -pass-remarks-analysis=pipeliner %s -filetype=null 2>&1 | FileCheck %s --check-prefix=RAISED

; This loop's MII is 32: two independent accumulators each issue one MFMA per
; iteration, and each MFMA holds the XDL pipe for 16 cycles (res=32). That
; exceeds the generic default cap of 27. At default, pipeliner aborts and
; only a raised cap lets it schedule.
; DEFAULT: Minimal Initiation Interval too large: 32 > 27
; RAISED: Schedule found with Initiation Interval

define amdgpu_kernel void @swp_amdgpu_pipeline_max_mii(i32 %arg, ptr addrspace(3) %p) {
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
