; RUN: opt -passes='require<profile-summary>,function(select-optimize)' -mtriple=aarch64-linux-gnu -mcpu=generic -S < %s | FileCheck %s

; Test that xor auxiliaries are not grouped into BinOp SelectLikes (#226417).

define void @not_xor_add_not_selectlike(i32 %n, i1 %c, i1 %v, i32 %a, i32 %b) {
; CHECK-LABEL: @not_xor_add_not_selectlike(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[OUTER:%.*]]
; CHECK:       outer:
; CHECK:       latch:
; CHECK-NEXT:    [[NC:%.*]] = xor i1 [[C:%.*]], true
; CHECK-NEXT:    [[SH:%.*]] = lshr i1 [[V:%.*]], 0
; CHECK-NEXT:    [[S0:%.*]] = select i1 [[C]], i32 [[A:%.*]], i32 [[B:%.*]]
; CHECK-NEXT:    [[ADD:%.*]] = add i1 [[NC]], [[SH]]
; CHECK-NEXT:    [[S2:%.*]] = select i1 [[C]], i32 [[A]], i32 [[B]]
; CHECK-NOT: select.end
; CHECK-NOT: freeze
;
entry:
  br label %outer

outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  br label %inner

inner:
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner ]
  %j.next = add i32 %j, 1
  %ic = icmp slt i32 %j.next, %n
  br i1 %ic, label %inner, label %latch

latch:
  %nc = xor i1 %c, true
  %sh = lshr i1 %v, 0
  %s0 = select i1 %c, i32 %a, i32 %b
  %add = add i1 %nc, %sh
  %s2 = select i1 %c, i32 %a, i32 %b
  %i.next = add i32 %i, 1
  %oc = icmp slt i32 %i.next, %n
  br i1 %oc, label %outer, label %exit

exit:
  ret void
}

define void @not_xor_sub_not_selectlike(i32 %n, i1 %c, i1 %v, i32 %a, i32 %b) {
; CHECK-LABEL: @not_xor_sub_not_selectlike(
; CHECK:       latch:
; CHECK-NEXT:    [[NC:%.*]] = xor i1 [[C:%.*]], true
; CHECK-NEXT:    [[SH:%.*]] = lshr i1 [[V:%.*]], 0
; CHECK-NEXT:    [[SUB:%.*]] = sub i1 [[SH]], [[NC]]
; CHECK-NEXT:    [[S0:%.*]] = select i1 [[C]], i32 [[A:%.*]], i32 [[B:%.*]]
; CHECK-NEXT:    [[S2:%.*]] = select i1 [[C]], i32 [[A]], i32 [[B]]
; CHECK-NOT: select.end
;
entry:
  br label %outer

outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  br label %inner

inner:
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner ]
  %j.next = add i32 %j, 1
  %ic = icmp slt i32 %j.next, %n
  br i1 %ic, label %inner, label %latch

latch:
  %nc = xor i1 %c, true
  %sh = lshr i1 %v, 0
  %sub = sub i1 %sh, %nc
  %s0 = select i1 %c, i32 %a, i32 %b
  %s2 = select i1 %c, i32 %a, i32 %b
  %i.next = add i32 %i, 1
  %oc = icmp slt i32 %i.next, %n
  br i1 %oc, label %outer, label %exit

exit:
  ret void
}
