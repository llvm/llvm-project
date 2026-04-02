; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

;             Alpha (div.uni)
;              |   \
;             Entry \
;          (div.cond)\
;             /   \   \
;            B0   B3  |
;            |    |   |
;            B1   B4<-+
;            |    |
;            B2   B5
;          /  |    |
;         /   |   B501
;        /    |    |
;     B201->B202  B502
;             \  /
;              B6 (phi: divergent)
;
;
; CHECK-LABEL:  'test_ctrl_divergence':
; CHECK-LABEL:  BLOCK Entry
; CHECK:  DIVERGENT:   %div.cond = icmp eq i32 %tid, 0
; CHECK:  DIVERGENT:   br i1 %div.cond, label %B3, label %B0
;
; CHECK-LABEL:  BLOCK B6
; CHECK:  DIVERGENT:   %div_a = phi i32 [ %a0, %B202 ], [ %a1, %B502 ]
; CHECK:  DIVERGENT:   %div_b = phi i32 [ %b0, %B202 ], [ %b1, %B502 ]
; CHECK:  DIVERGENT:   %div_c = phi i32 [ %c0, %B202 ], [ %c1, %B502 ]

define amdgpu_kernel void @test_ctrl_divergence(i32 %a, i32 %b, i32 %c, i32 %d) {
Alpha:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %div.uni = icmp eq i32 %a, 0
  br i1 %div.uni, label %Entry, label %B4

Entry:
  %div.cond = icmp eq i32 %tid, 0
  br i1 %div.cond, label %B3, label %B0 ; divergent branch

B0:
  br label %B1

B1:
  br label %B2

B2:
  %a0 = add i32 %a, 1
  %b0 = add i32 %b, 2
  %c0 = add i32 %c, 3
  br i1 %div.uni, label %B201, label %B202

B201:
  br label %B202

B202:
  br label %B6

B3:
  br label %B4

B4:
  %a1 = add i32 %a, 10
  %b1 = add i32 %b, 20
  %c1 = add i32 %c, 30
  br i1 %div.uni, label %B5, label %B501

B5:
  br label %B501

B501:
  br label %B502

B502:
  br label %B6

B6:
  %div_a = phi i32 [%a0, %B202], [%a1,  %B502]
  %div_b = phi i32 [%b0, %B202], [%b1,  %B502]
  %div_c = phi i32 [%c0, %B202], [%c1,  %B502]
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = {nounwind readnone }
