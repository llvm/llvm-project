; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: BB0_2
; GCN-NOT: v_readfirstlane


target triple = "amdgcn--amdhsa"
define amdgpu_kernel void @uniform-PHI(i32 addrspace(1)* nocapture readonly %arg, i32 addrspace(1)* nocapture %arg1, i32 %arg2) {
bb:
  %tmp = sext i32 %arg2 to i64
  %tmp3 = tail call i64 @_Z13get_global_idj(i32 0) #2
  %tmp4 = icmp ugt i64 %tmp3, %tmp
  %tmp5 = icmp sgt i32 %arg2, 0
  %tmp6 = and i1 %tmp4, %tmp5
  br i1 %tmp6, label %bb7, label %bb17

bb7:                                              ; preds = %bb
  br label %bb8

bb8:                                              ; preds = %bb8, %bb7
  %tmp9 = phi i32 [ %tmp15, %bb8 ], [ 0, %bb7 ]
  %tmp10 = phi i32 [ %tmp14, %bb8 ], [ 0, %bb7 ]
  %tmp11 = zext i32 %tmp9 to i64
  %tmp12 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp11
  %tmp13 = load i32, i32 addrspace(1)* %tmp12, align 4
  %tmp14 = add nsw i32 %tmp13, %tmp10
  %tmp15 = add nuw nsw i32 %tmp9, 1
  %tmp16 = icmp eq i32 %tmp15, %arg2
  br i1 %tmp16, label %bb17, label %bb8

bb17:                                             ; preds = %bb8, %bb
  %tmp18 = phi i32 [ 0, %bb ], [ %tmp14, %bb8 ]
  store i32 %tmp18, i32 addrspace(1)* %arg1, align 4
  ret void
}

declare i64 @_Z13get_global_idj(i32) local_unnamed_addr #1
attributes #1 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="fiji" "target-features"="+16-bit-insts,+dpp,+fp64-fp16-denormals,+s-memrealtime,-fp32-denormals" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind readnone }
