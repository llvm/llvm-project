; RUN: opt -opaque-pointers=0 -S -mtriple=amdgcn-amd-amdhsa -loop-reduce %s | FileCheck %s

; Test for assert resulting from inconsistent isLegalAddressingMode
; answers when the address space was dropped from the query.

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

%0 = type { i32, double, i32, float }

; CHECK-LABEL: @lsr_crash_preserve_addrspace_unknown_type(
; CHECK: %tmp4 = bitcast %0 addrspace(3)* %tmp to double addrspace(3)*
; CHECK: %scevgep5 = getelementptr double, double addrspace(3)* %tmp4, i32 1
; CHECK: load double, double addrspace(3)* %scevgep5

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

; CHECK-LABEL: @lsr_crash_preserve_addrspace_unknown_type2(
; CHECK: %scevgep3 = getelementptr i8, i8 addrspace(5)* %array, i32 %j
; CHECK: %scevgep2 = getelementptr i8, i8 addrspace(5)* %array, i32 %j
; CHECK: %n8 = load i8, i8 addrspace(5)* %scevgep2, align 4
; CHECK: call void @llvm.memcpy.p5i8.p3i8.i64(i8 addrspace(5)* %scevgep3, i8 addrspace(3)* %scevgep4, i64 42, i1 false)
; CHECK: call void @llvm.memmove.p5i8.p3i8.i64(i8 addrspace(5)* %scevgep3, i8 addrspace(3)* %scevgep4, i64 42, i1 false)
; CHECK: call void @llvm.memset.p5i8.i64(i8 addrspace(5)* %scevgep3, i8 42, i64 42, i1 false)
define void @lsr_crash_preserve_addrspace_unknown_type2(i8 addrspace(5)* %array, i8 addrspace(3)* %array2) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %j = phi i32 [ %add, %for.inc ], [ 0, %entry ]
  %idx = getelementptr inbounds i8, i8 addrspace(5)* %array, i32 %j
  %idx1 = getelementptr inbounds i8, i8 addrspace(3)* %array2, i32 %j
  %t = getelementptr inbounds i8, i8 addrspace(5)* %array, i32 %j
  %n8 = load i8, i8 addrspace(5)* %t, align 4
  %n7 = getelementptr inbounds i8, i8 addrspace(5)* %t, i32 42
  %n9 = load i8, i8 addrspace(5)* %n7, align 4
  %cmp = icmp sgt i32 %j, 42
  %add = add nuw nsw i32 %j, 1
  br i1 %cmp, label %if.then17, label %for.inc

if.then17:                                        ; preds = %for.body
  call void @llvm.memcpy.p5i8.p5i8.i64(i8 addrspace(5)* %idx, i8 addrspace(3)* %idx1, i64 42, i1 false)
  call void @llvm.memmove.p5i8.p5i8.i64(i8 addrspace(5)* %idx, i8 addrspace(3)* %idx1, i64 42, i1 false)
  call void @llvm.memset.p5i8.i64(i8 addrspace(5)* %idx, i8 42, i64 42, i1 false)
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then17
  %exitcond = icmp eq i1 %cmp, 1
  br i1 %exitcond, label %end, label %for.body

end:                                              ; preds = %for.inc
  ret void
}

declare void @llvm.memcpy.p5i8.p5i8.i64(i8 addrspace(5)*, i8 addrspace(3)*, i64, i1)
declare void @llvm.memmove.p5i8.p5i8.i64(i8 addrspace(5)*, i8 addrspace(3)*, i64, i1)
declare void @llvm.memset.p5i8.i64(i8 addrspace(5)*, i8, i64, i1)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
