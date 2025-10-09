;
; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s
;
;
;      Entry (div.cond)
;      /   \
;     B0   B3
;     |    |
;     B1   B4
;     |    |
;      \  /
;       B5 (phi: divergent)
;       |
;       B6  (div.uni)
;      /   \
;     B7   B9
;     |    |
;     B8   B10
;     |    |
;      \  /
;       B11 (phi: uniform)


; CHECK-LABEL:  'test_ctrl_divergence':
; CHECK-LABEL:  BLOCK Entry
; CHECK:  DIVERGENT:   %div.cond = icmp eq i32 %tid, 0
; CHECK:  DIVERGENT:   br i1 %div.cond, label %B3, label %B0
;
; CHECK-LABEL:  BLOCK B5
; CHECK:  DIVERGENT:   %div_a = phi i32 [ %a0, %B1 ], [ %a1, %B4 ]
; CHECK:  DIVERGENT:   %div_b = phi i32 [ %b0, %B1 ], [ %b1, %B4 ]
;
; CHECK-LABEL:  BLOCK B6
; CHECK-NOT:  DIVERGENT:   %uni.cond = icmp
; CHECK-NOT:  DIVERGENT:   br i1 %div.cond
;
; CHECK-LABEL:  BLOCK B11
; CHECK-NOT:  DIVERGENT:   %div_d = phi i32


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
  br label %B5

B3:
  %a1 = add i32 %a, 10
  br label %B4

B4:
  %b1 = add i32 %b, 20
  br label %B5

B5:
  %div_a = phi i32 [%a0, %B1], [%a1,  %B4]
  %div_b = phi i32 [%b0, %B1], [%b1,  %B4]
  br label %B6

B6:
  %uni.cond = icmp eq i32 %c, 0
  br i1 %uni.cond, label %B7, label %B9

B7:
  %d1 = add i32 %d, 1
  br label %B8

B8:
  br label %B11

B9:
  %d2 = add i32 %d, 3
  br label %B10

B10:
  br label %B11

B11:
  %div_d = phi i32 [%d1, %B8], [%d2, %B10]
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = {nounwind readnone }
