; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s
;
; This is to test an if-then-else case with some unmerged basic blocks
; (https://github.com/llvm/llvm-project/issues/137277)
;
;      Entry (div.cond)
;      /   \
;     B0   B3
;     |    |
;     B1   B4
;     |    |
;     B2   B5
;      \  /
;       B6 (phi: divergent)
;


; CHECK-LABEL:  'test_ctrl_divergence':
; CHECK-LABEL:  BLOCK Entry
; CHECK:  DIVERGENT:   %div.cond = icmp eq i32 %tid, 0
; CHECK:  DIVERGENT:   br i1 %div.cond, label %B3, label %B0
;
; CHECK-LABEL:  BLOCK B6
; CHECK:  DIVERGENT:   %div_a = phi i32 [ %a0, %B2 ], [ %a1, %B5 ]
; CHECK:  DIVERGENT:   %div_b = phi i32 [ %b0, %B2 ], [ %b1, %B5 ]
; CHECK:  DIVERGENT:   %div_c = phi i32 [ %c0, %B2 ], [ %c1, %B5 ]


define amdgpu_kernel void @test_ctrl_divergence(i32 %a, i32 %b, i32 %c, i32 %d) {
Entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %B3, label %B0 ; divergent branch

B0:
  %a0 = add i32 %a, 1
  br label %B1

B1:
  %b0 = add i32 %b, 2
  br label %B2

B2:
  %c0 = add i32 %c, 3
  br label %B6

B3:
  %a1 = add i32 %a, 10
  br label %B4

B4:
  %b1 = add i32 %b, 20
  br label %B5

B5:
  %c1 = add i32 %c, 30
  br label %B6

B6:
  %div_a = phi i32 [%a0, %B2], [%a1,  %B5]
  %div_b = phi i32 [%b0, %B2], [%b1,  %B5]
  %div_c = phi i32 [%c0, %B2], [%c1,  %B5]
  br i1 %div.cond, label %B8, label %B7 ; divergent branch

B7:
  %d1 = add i32 %d, 1
  br label %B8

B8:
  %div_d = phi i32 [%d1, %B7], [%d, %B6]
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x()
