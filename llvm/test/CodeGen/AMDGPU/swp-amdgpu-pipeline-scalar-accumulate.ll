; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -amdgpu-enable-pipeliner -pass-remarks-analysis=pipeliner %s -filetype=null 2>&1 | FileCheck %s
; Reduced from Composable Kernels. Scalar fp reduction: an i16 buffer
; load feeding a transcendental exp() with a NaN-guarded accumulate. Verifies
; such a loop software-pipelines.
; CHECK: Schedule found with Initiation Interval

define amdgpu_kernel void @swp_amdgpu_pipeline_scalar_accumulate(i1 %arg) {
bb:
  %call = tail call i32 @llvm.amdgcn.workitem.id.x()
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %phi = phi float [ 0.000000e+00, %bb ], [ %select6, %bb1 ]
  %phi2 = phi i32 [ 0, %bb ], [ 1, %bb1 ]
  %phi3 = phi i32 [ %call, %bb ], [ %add, %bb1 ]
  %add = add i32 %phi3, -8
  %call4 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %phi2, i32 0, i32 0)
  %icmp = icmp slt i32 %add, 0
  %and = and i1 %arg, %icmp
  %bitcast = bitcast i16 %call4 to half
  %fpext = fpext half %bitcast to float
  %select = select i1 %and, float %fpext, float 0.000000e+00
  %fsub = fsub float %select, 1.000000e+00
  %call5 = tail call float @llvm.exp.f32(float %fsub)
  %fcmp = fcmp uno float %call5, 0.000000e+00
  %fadd = fadd float %phi, 1.000000e+00
  %select6 = select i1 %fcmp, float %phi, float %fadd
  %icmp7 = icmp eq i32 %phi2, 0
  br i1 %icmp7, label %bb8, label %bb1

bb8:                                              ; preds = %bb1
  ret void
}
