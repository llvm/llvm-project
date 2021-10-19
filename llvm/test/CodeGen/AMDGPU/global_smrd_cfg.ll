; RUN: llc -mtriple amdgcn--amdhsa -mcpu=fiji -amdgpu-scalarize-global-loads=true -verify-machineinstrs  < %s | FileCheck %s

; CHECK-LABEL: %bb22

; Load from %arg has alias store in Loop

; CHECK: flat_load_dword

; #####################################################################

; Load from %arg1 has no-alias store in Loop - arg1[i+1] never alias arg1[i]
; However, our analysis cannot detect this.

; CHECK: flat_load_dword

; #####################################################################

; CHECK-LABEL: %bb11

; Load from %arg in a Loop body has alias store

; CHECK: flat_load_dword

; CHECK-LABEL: %bb20

; CHECK: flat_store_dword

define amdgpu_kernel void @cfg(i32 addrspace(1)* nocapture readonly %arg, i32 addrspace(1)* nocapture %arg1, i32 %arg2) #0 {
bb:
  %tmp = sext i32 %arg2 to i64
  %tmp3 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp
  %tmp4 = load i32, i32 addrspace(1)* %tmp3, align 4, !tbaa !0
  %tmp5 = icmp sgt i32 %tmp4, 0
  br i1 %tmp5, label %bb6, label %bb8

bb6:                                              ; preds = %bb
  br label %bb11

bb7:                                              ; preds = %bb22
  br label %bb8

bb8:                                              ; preds = %bb7, %bb
  %tmp9 = phi i32 [ 0, %bb ], [ %tmp30, %bb7 ]
  %tmp10 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp
  store i32 %tmp9, i32 addrspace(1)* %tmp10, align 4, !tbaa !0
  ret void

bb11:                                             ; preds = %bb22, %bb6
  %tmp12 = phi i32 [ %tmp30, %bb22 ], [ 0, %bb6 ]
  %tmp13 = phi i32 [ %tmp25, %bb22 ], [ 0, %bb6 ]
  %tmp14 = srem i32 %tmp13, %arg2
  %tmp15 = sext i32 %tmp14 to i64
  %tmp16 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp15
  %tmp17 = load i32, i32 addrspace(1)* %tmp16, align 4, !tbaa !0
  %tmp18 = icmp sgt i32 %tmp17, 100
  %tmp19 = sext i32 %tmp13 to i64
  br i1 %tmp18, label %bb20, label %bb22

bb20:                                             ; preds = %bb11
  %tmp21 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp19
  store i32 0, i32 addrspace(1)* %tmp21, align 4, !tbaa !0
  br label %bb22

bb22:                                             ; preds = %bb20, %bb11
  %tmp23 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp19
  %tmp24 = load i32, i32 addrspace(1)* %tmp23, align 4, !tbaa !0
  %tmp25 = add nuw nsw i32 %tmp13, 1
  %tmp26 = sext i32 %tmp25 to i64
  %tmp27 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp26
  %tmp28 = load i32, i32 addrspace(1)* %tmp27, align 4, !tbaa !0
  %tmp29 = add i32 %tmp24, %tmp12
  %tmp30 = add i32 %tmp29, %tmp28
  %tmp31 = icmp eq i32 %tmp25, %tmp4
  br i1 %tmp31, label %bb7, label %bb11
}

; one more test to ensure that aliasing store after the load
; is considered clobbering if load parent block is the same 
; as a loop header block.

; CHECK-LABEL: %bb1

; Load from %arg has alias store that is after the load 
; but is considered clobbering because of the loop.

; CHECK: flat_load_dword

define amdgpu_kernel void @cfg_selfloop(i32 addrspace(1)* nocapture readonly %arg, i32 addrspace(1)* nocapture %arg1, i32 %arg2) #0 {
bb:
  br label %bb1

bb2:
  ret void

bb1:
  %tmp13 = phi i32 [ %tmp25, %bb1 ], [ 0, %bb ]
  %tmp14 = srem i32 %tmp13, %arg2
  %tmp15 = sext i32 %tmp14 to i64
  %tmp16 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp15
  %tmp17 = load i32, i32 addrspace(1)* %tmp16, align 4, !tbaa !0
  %tmp19 = sext i32 %tmp13 to i64
  %tmp21 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp19
  store i32 %tmp17, i32 addrspace(1)* %tmp21, align 4, !tbaa !0
  %tmp25 = add nuw nsw i32 %tmp13, 1
  %tmp31 = icmp eq i32 %tmp25, 100
  br i1 %tmp31, label %bb2, label %bb1
}


attributes #0 = { "target-cpu"="fiji" }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
