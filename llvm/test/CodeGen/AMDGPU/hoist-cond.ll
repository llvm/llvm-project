; XFAIL: *
; RUN: llc -mtriple=amdgcn -verify-machineinstrs -disable-block-placement < %s | FileCheck %s

; Check that invariant compare is hoisted out of the loop.
; At the same time condition shall not be serialized into a VGPR and deserialized later
; using another v_cmp + v_cndmask, but used directly in s_and_saveexec_b64.

; CHECK: v_cmp_{{..}}_u32_e{{32|64}} [[COND:s\[[0-9]+:[0-9]+\]|vcc]]
; CHECK: BB0_1:
; CHECK-NOT: v_cmp
; CHECK-NOT: v_cndmask
; CHECK: s_and_saveexec_b64 s[{{[0-9]+:[0-9]+}}], [[COND]]
; CHECK: ; %bb.2:

define amdgpu_kernel void @hoist_cond(ptr addrspace(1) nocapture %arg, ptr addrspace(1) noalias nocapture readonly %arg1, i32 %arg3, i32 %arg4) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tmp5 = icmp ult i32 %tmp, %arg3
  br label %bb1

bb1:                                              ; preds = %bb3, %bb
  %tmp7 = phi i32 [ %arg4, %bb ], [ %tmp16, %bb3 ]
  %tmp8 = phi float [ 0.000000e+00, %bb ], [ %tmp15, %bb3 ]
  br i1 %tmp5, label %bb2, label %bb3

bb2:                                              ; preds = %bb1
  %tmp10 = zext i32 %tmp7 to i64
  %tmp11 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 %tmp10
  %tmp12 = load float, ptr addrspace(1) %tmp11, align 4
  br label %bb3

bb3:                                             ; preds = %bb2, %bb1
  %tmp14 = phi float [ %tmp12, %bb2 ], [ 0.000000e+00, %bb1 ]
  %tmp15 = fadd float %tmp8, %tmp14
  %tmp16 = add i32 %tmp7, -1
  %tmp17 = icmp eq i32 %tmp16, 0
  br i1 %tmp17, label %bb4, label %bb1

bb4:                                             ; preds = %bb3
  store float %tmp15, ptr addrspace(1) %arg, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
