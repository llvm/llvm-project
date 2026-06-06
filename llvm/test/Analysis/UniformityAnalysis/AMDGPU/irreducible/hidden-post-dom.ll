; RUN: opt %s -mtriple amdgcn-- -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s

define amdgpu_kernel void @cycle_inner_ipd(i32 %n, i32 %a, i32 %b) #0 {
;
;          entry
;        /      \
;      E2<------E1
;       | \     ^^
;       |  \  /  |
;       |   v/   |
;       |   A    |
;       |  /     |
;       | /      |
;       vv       |
;       B------->C
;                |
;                X
;
;
; CHECK-LABEL: BLOCK entry
; CHECK:  DIVERGENT:   %tid = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK:  DIVERGENT:   %div.cond = icmp slt i32 %tid, 0
; CHECK: END BLOCK
;
; CHECK-LABEL: BLOCK B
; CHECK:  DIVERGENT:   %div.merge = phi i32 [ 0, %A ], [ %b, %E2 ]
; CHECK: END BLOCK

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp slt i32 %tid, 0
  %uni.cond = icmp slt i32 %a, 0
  %uni.cond1 = icmp slt i32 %a, 2
  %uni.cond2 = icmp slt i32 %a, 10
  br i1 %uni.cond, label %E2, label %E1

E1:
  br label %E2

E2:
  br i1 %uni.cond1, label %A, label %B


A:
  br i1 %div.cond, label %E1, label %B

B:
  %div.merge = phi i32 [ 0, %A ], [ %b, %E2 ]
  br label %C

C:
  br i1 %uni.cond2, label %E1, label %X

X:
  ret void
}
