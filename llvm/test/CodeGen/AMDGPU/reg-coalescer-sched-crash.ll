; RUN: llc -march=amdgcn -verify-machineinstrs -o /dev/null < %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs -o /dev/null < %s

; The register coalescer introduces a verifier error which later
; results in a crash during scheduling.

declare i32 @llvm.amdgcn.workitem.id.x() #0

define amdgpu_kernel void @reg_coalescer_breaks_dead(ptr addrspace(1) nocapture readonly %arg, i32 %arg1, i32 %arg2, i32 %arg3) #1 {
bb:
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  %cmp0 = icmp eq i32 %id.x, 0
  br i1 %cmp0, label %bb3, label %bb4

bb3:                                              ; preds = %bb
  %tmp = ashr exact i32 undef, 8
  br label %bb6

bb4:                                              ; preds = %bb6, %bb
  %tmp5 = phi <2 x i32> [ zeroinitializer, %bb ], [ %tmp13, %bb6 ]
  br i1 undef, label %bb15, label %bb16

bb6:                                              ; preds = %bb6, %bb3
  %tmp7 = phi <2 x i32> [ zeroinitializer, %bb3 ], [ %tmp13, %bb6 ]
  %tmp8 = add nsw i32 0, %arg1
  %tmp9 = add nsw i32 %tmp8, 0
  %tmp10 = sext i32 %tmp9 to i64
  %tmp11 = getelementptr inbounds <2 x i32>, ptr addrspace(1) %arg, i64 %tmp10
  %tmp12 = load <2 x i32>, ptr addrspace(1) %tmp11, align 8
  %tmp13 = add <2 x i32> %tmp12, %tmp7
  %tmp14 = icmp slt i32 undef, %arg2
  br i1 %tmp14, label %bb6, label %bb4

bb15:                                             ; preds = %bb4
  store <2 x i32> %tmp5, ptr addrspace(3) undef, align 8
  br label %bb16

bb16:                                             ; preds = %bb15, %bb4
  unreachable
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
