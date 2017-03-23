; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -loop-reduce %s | FileCheck %s

; Test for assert resulting from inconsistent isLegalAddressingMode
; answers when the address space was dropped from the query.

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"

%0 = type { i32, double, i32, float }

; CHECK-LABEL: @lsr_crash_preserve_addrspace_unknown_type(
; CHECK: %tmp4 = bitcast %0 addrspace(3)* %tmp to double addrspace(3)*
; CHECK: %scevgep5 = getelementptr double, double addrspace(3)* %tmp4, i32 1
; CHEC: load double, double addrspace(3)* %scevgep5

; CHECK: %scevgep = getelementptr i32, i32 addrspace(3)* %tmp1, i32 4
; CHECK:%tmp14 = load i32, i32 addrspace(3)* %scevgep
define amdgpu_kernel void @lsr_crash_preserve_addrspace_unknown_type() #0 {
bb:
  br label %bb1

bb1:                                              ; preds = %bb17, %bb
  %tmp = phi %0 addrspace(3)* [ undef, %bb ], [ %tmp18, %bb17 ]
  %tmp2 = getelementptr inbounds %0, %0 addrspace(3)* %tmp, i64 0, i32 1
  %tmp3 = load double, double addrspace(3)* %tmp2, align 8
  br label %bb4

bb4:                                              ; preds = %bb1
  br i1 undef, label %bb8, label %bb5

bb5:                                              ; preds = %bb4
  unreachable

bb8:                                              ; preds = %bb4
  %tmp9 = getelementptr inbounds %0, %0 addrspace(3)* %tmp, i64 0, i32 0
  %tmp10 = load i32, i32 addrspace(3)* %tmp9, align 4
  %tmp11 = icmp eq i32 0, %tmp10
  br i1 %tmp11, label %bb12, label %bb17

bb12:                                             ; preds = %bb8
  %tmp13 = getelementptr inbounds %0, %0 addrspace(3)* %tmp, i64 0, i32 2
  %tmp14 = load i32, i32 addrspace(3)* %tmp13, align 4
  %tmp15 = icmp eq i32 0, %tmp14
  br i1 %tmp15, label %bb16, label %bb17

bb16:                                             ; preds = %bb12
  unreachable

bb17:                                             ; preds = %bb12, %bb8
  %tmp18 = getelementptr inbounds %0, %0 addrspace(3)* %tmp, i64 2
  br label %bb1
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
