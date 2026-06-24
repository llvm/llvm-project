; RUN: opt -S -passes=gvn < %s | FileCheck %s --check-prefix=GVN
; RUN: opt -S -passes='gvn,dse' < %s | FileCheck %s --check-prefix=DSE

define i16 @test(ptr %p1, ptr %p2) {
; GVN-LABEL: define i16 @test(
; GVN-SAME: ptr [[P1:%.*]], ptr [[P2:%.*]]) {
; GVN:       entry:
; GVN-NEXT:    store i8 0, ptr [[P1]], align 1, !noalias [[SCOPE:![0-9]+]]
; GVN-NEXT:    [[L16:%.*]] = load i16, ptr [[P2]], align 2, !alias.scope [[SCOPE]]
; GVN-NEXT:    [[L8:%.*]] = trunc i16 [[L16]] to i8
; GVN-NEXT:    [[COND:%.*]] = icmp eq i8 [[L8]], 7
; GVN-NEXT:    store i8 1, ptr [[P1]], align 1
; GVN-NEXT:    [[R:%.*]] = select i1 [[COND]], i16 [[L16]], i16 0
; GVN-NEXT:    ret i16 [[R]]
;
; DSE-LABEL: define i16 @test(
; DSE-SAME: ptr [[P1:%.*]], ptr [[P2:%.*]]) {
; DSE:       entry:
; DSE-NEXT:    [[L16:%.*]] = load i16, ptr [[P2]], align 2, !alias.scope [[SCOPE:![0-9]+]]
; DSE-NEXT:    [[L8:%.*]] = trunc i16 [[L16]] to i8
; DSE-NEXT:    [[COND:%.*]] = icmp eq i8 [[L8]], 7
; DSE-NEXT:    store i8 1, ptr [[P1]], align 1
; DSE-NEXT:    [[R:%.*]] = select i1 [[COND]], i16 [[L16]], i16 0
; DSE-NEXT:    ret i16 [[R]]
;
entry:
  store i8 0, ptr %p1, align 1, !noalias !2
  %l16 = load i16, ptr %p2, align 2, !alias.scope !2
  %l8 = load i8, ptr %p2, align 1, !alias.scope !2
  %cond = icmp eq i8 %l8, 7
  store i8 1, ptr %p1, align 1
  %r = select i1 %cond, i16 %l16, i16 0
  ret i16 %r
}

!0 = !{!0}
!1 = !{!1, !0}
!2 = !{!1}
