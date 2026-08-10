; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s
;
; This is to test a divergent phi involving loops
; (https://github.com/llvm/llvm-project/issues/137277).
;
;        B0 (div.cond)
;      /   \
;  (L)B1   B4
;     |    |
;     B2   B5 (L)
;     |    |
;     B3   /
;      \  /
;      B6 (phi: divergent)
;

;
; CHECK-LABEL: UniformityInfo for function 'test_loop_ctrl_divergence':
; CHECK-LABEL: BLOCK Entry
; CHECK: DIVERGENT:   %tid = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-LABEL: BLOCK B0
; CHECK: DIVERGENT:   %div.cond = icmp eq i32 %tid, 0
; CHECK-LABEL: BLOCK B3
; CHECK: %uni_a = phi i32 [ %a1, %B2 ], [ %a, %Entry ]
; CHECK-LABEL: BLOCK B5
; CHECK: %uni.a3 = phi i32 [ %a2, %B4 ], [ %uni_a3, %B5 ]
; CHECK-LABEL BLOCK B6
; CHECK: DIVERGENT:   %div_a = phi i32 [ %uni_a, %B3 ], [ %uni_a3, %B5 ]
;

define amdgpu_kernel void @test_loop_ctrl_divergence(i32 %a, i32 %b, i32 %c, i32 %d) {
Entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %uni.cond0 = icmp eq i32 %d, 0
  br i1 %uni.cond0, label %B3, label %B0 ; uniform branch

B0:
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %B4, label %B1 ; divergent branch

B1:
  %uni.a0 = phi i32 [%a, %B0], [%a0, %B1]
  %a0 = add i32 %uni.a0, 1
  %uni.cond1 = icmp slt i32 %a0, %b
  br i1 %uni.cond1, label %B1, label %B2

B2:
  %a1 = add i32 %a0, 10
  br label %B3

B3:
  %uni_a = phi i32 [%a1, %B2], [%a,  %Entry]
  br label %B6

B4:
  %a2 = add i32 %a, 20
  br label %B5

B5:
  %uni.a3= phi i32 [%a2, %B4], [%uni_a3, %B5]
  %uni_a3 = add i32 %uni.a3, 1
  %uni.cond2 = icmp slt i32 %uni_a3, %c
  br i1 %uni.cond2, label %B5, label %B6

B6:
  %div_a = phi i32 [%uni_a, %B3], [%uni_a3, %B5] ;   divergent
  %div.cond2 = icmp eq i32 %tid, 2
  br i1 %div.cond2, label %B7, label %B8 ; divergent branch

B7:
  %c0 = add i32 %div_a, 2 ; divergent
  br label %B8

B8:
  %ret = phi i32 [%c0, %B7], [0, %B6] ; divergent
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
