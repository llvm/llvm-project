; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -amdgpu-enable-pipeliner -debug-only=pipeliner %s -filetype=null 2>&1 | FileCheck %s
; REQUIRES: asserts

; Resource-bound II set by the aggregate micro-op issue width, not by any single
; functional unit. The body issues 14 single-micro-op instructions across three
; unit classes (VALU=6, SALU=5, VMEM=3), so no one unit reaches 14, but at an
; issue width of 1 the 14 micro-ops force res=14. That dominates the short
; recurrence (rec=3), so the II is resource-bound on issue, not recurrence-bound.
; CHECK: MII = 14 MAX_II = 24 (rec=3, res=14)

define amdgpu_kernel void @swp_amdgpu_pipeline_resmii_issue_bound(i1 %arg, ptr addrspace(1) %out) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %iv = phi i32 [ 0, %bb ], [ %iv.next, %bb1 ]
  %acc = phi float [ 0.000000e+00, %bb ], [ %fadd3, %bb1 ]
  %off0 = shl i32 %iv, 2
  %off1 = or i32 %off0, 1
  %off2 = or i32 %off0, 2
  %load0 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) null, i32 %off0, i32 0, i32 0)
  %load1 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) null, i32 %off1, i32 0, i32 0)
  %load2 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) null, i32 %off2, i32 0, i32 0)
  %f0 = bitcast i32 %load0 to float
  %f1 = bitcast i32 %load1 to float
  %f2 = bitcast i32 %load2 to float
  %sel0 = select i1 %arg, float %f0, float 0.000000e+00
  %sel1 = select i1 %arg, float %f1, float 0.000000e+00
  %sel2 = select i1 %arg, float %f2, float 0.000000e+00
  %fadd1 = fadd float %acc, %sel0
  %fadd2 = fadd float %fadd1, %sel1
  %fadd3 = fadd float %fadd2, %sel2
  %iv.next = add i32 %iv, 1
  %icmp = icmp ult i32 %iv.next, 100
  br i1 %icmp, label %bb1, label %exit

exit:                                             ; preds = %bb1
  %lcssa = phi float [ %fadd3, %bb1 ]
  store float %lcssa, ptr addrspace(1) %out, align 4
  ret void
}
