; RUN: opt -S -correlated-propagation < %s | FileCheck %s

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)

declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32)

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)

declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32)

declare void @llvm.trap()


define i32 @signed_add(i32 %x, i32 %y) {
; CHECK-LABEL: @signed_add(
; CHECK: @llvm.ssub.with.overflow.i32
; CHECK: @llvm.ssub.with.overflow.i32
; CHECK: @llvm.sadd.with.overflow.i32
entry:
  %cmp = icmp sgt i32 %y, 0
  br i1 %cmp, label %land.lhs.true, label %lor.lhs.false

land.lhs.true:                                    ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 2147483647, i32 %y)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:                                             ; preds = %land.lhs.true, %land.lhs.true3, %cond.false
  tail call void @llvm.trap()
  unreachable

cont:                                             ; preds = %land.lhs.true
  %2 = extractvalue { i32, i1 } %0, 0
  %cmp1 = icmp slt i32 %2, %x
  br i1 %cmp1, label %cond.end, label %cond.false

lor.lhs.false:                                    ; preds = %entry
  %cmp2 = icmp slt i32 %y, 0
  br i1 %cmp2, label %land.lhs.true3, label %cond.false

land.lhs.true3:                                   ; preds = %lor.lhs.false
  %3 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 -2147483648, i32 %y)
  %4 = extractvalue { i32, i1 } %3, 1
  br i1 %4, label %trap, label %cont4

cont4:                                            ; preds = %land.lhs.true3
  %5 = extractvalue { i32, i1 } %3, 0
  %cmp5 = icmp sgt i32 %5, %x
  br i1 %cmp5, label %cond.end, label %cond.false

cond.false:                                       ; preds = %cont, %cont4, %lor.lhs.false
  %6 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %x, i32 %y)
  %7 = extractvalue { i32, i1 } %6, 0
  %8 = extractvalue { i32, i1 } %6, 1
  br i1 %8, label %trap, label %cond.end

cond.end:                                         ; preds = %cond.false, %cont, %cont4
  %cond = phi i32 [ 0, %cont4 ], [ 0, %cont ], [ %7, %cond.false ]
  ret i32 %cond
}

define i32 @unsigned_add(i32 %x, i32 %y) {
; CHECK-LABEL: @unsigned_add(
; CHECK: @llvm.usub.with.overflow.i32
; CHECK: @llvm.uadd.with.overflow.i32
entry:
  %0 = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 -1, i32 %y)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:                                             ; preds = %cond.false, %entry
  tail call void @llvm.trap()
  unreachable

cont:                                             ; preds = %entry
  %2 = extractvalue { i32, i1 } %0, 0
  %cmp1 = icmp ult i32 %2, %x
  br i1 %cmp1, label %cond.end, label %cond.false

cond.false:                                       ; preds = %cont
  %3 = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %x, i32 %y)
  %4 = extractvalue { i32, i1 } %3, 0
  %5 = extractvalue { i32, i1 } %3, 1
  br i1 %5, label %trap, label %cond.end

cond.end:                                         ; preds = %cond.false, %cont
  %cond = phi i32 [ 0, %cont ], [ %4, %cond.false ]
  ret i32 %cond
}

define i32 @signed_sub(i32 %x, i32 %y) {
; CHECK-LABEL: @signed_sub(
; CHECK-NOT: @llvm.sadd.with.overflow.i32
; CHECK: @llvm.ssub.with.overflow.i32
entry:
  %cmp = icmp slt i32 %y, 0
  br i1 %cmp, label %land.lhs.true, label %lor.lhs.false

land.lhs.true:                                    ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %y, i32 2147483647)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:                                             ; preds = %land.lhs.true, %land.lhs.true3, %cond.false
  tail call void @llvm.trap()
  unreachable

cont:                                             ; preds = %land.lhs.true
  %2 = extractvalue { i32, i1 } %0, 0
  %cmp1 = icmp slt i32 %2, %x
  br i1 %cmp1, label %cond.end, label %cond.false

lor.lhs.false:                                    ; preds = %entry
  %cmp2 = icmp eq i32 %y, 0
  br i1 %cmp2, label %cond.false, label %land.lhs.true3

land.lhs.true3:                                   ; preds = %lor.lhs.false
  %3 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %y, i32 -2147483648)
  %4 = extractvalue { i32, i1 } %3, 1
  br i1 %4, label %trap, label %cont4

cont4:                                            ; preds = %land.lhs.true3
  %5 = extractvalue { i32, i1 } %3, 0
  %cmp5 = icmp sgt i32 %5, %x
  br i1 %cmp5, label %cond.end, label %cond.false

cond.false:                                       ; preds = %lor.lhs.false, %cont, %cont4
  %6 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %x, i32 %y)
  %7 = extractvalue { i32, i1 } %6, 0
  %8 = extractvalue { i32, i1 } %6, 1
  br i1 %8, label %trap, label %cond.end

cond.end:                                         ; preds = %cond.false, %cont, %cont4
  %cond = phi i32 [ 0, %cont4 ], [ 0, %cont ], [ %7, %cond.false ]
  ret i32 %cond
}

define i32 @unsigned_sub(i32 %x, i32 %y) {
; CHECK-LABEL: @unsigned_sub(
; CHECK: @llvm.usub.with.overflow.i32
entry:
  %cmp = icmp ult i32 %x, %y
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %x, i32 %y)
  %1 = extractvalue { i32, i1 } %0, 0
  %2 = extractvalue { i32, i1 } %0, 1
  br i1 %2, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %1, %cond.false ]
  ret i32 %cond
}

define i32 @signed_add_r1(i32 %x) {
; CHECK-LABEL: @signed_add_r1(
; CHECK-NOT: @llvm.sadd.with.overflow.i32
entry:
  %cmp = icmp eq i32 %x, 2147483647
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %x, i32 1)
  %1 = extractvalue { i32, i1 } %0, 0
  %2 = extractvalue { i32, i1 } %0, 1
  br i1 %2, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %1, %cond.false ]
  ret i32 %cond
}

define i32 @unsigned_add_r1(i32 %x) {
; CHECK-LABEL: @unsigned_add_r1(
; CHECK-NOT: @llvm.uadd.with.overflow.i32
entry:
  %cmp = icmp eq i32 %x, -1
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %x, i32 1)
  %1 = extractvalue { i32, i1 } %0, 0
  %2 = extractvalue { i32, i1 } %0, 1
  br i1 %2, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %1, %cond.false ]
  ret i32 %cond
}

define i32 @signed_sub_r1(i32 %x) {
; CHECK-LABEL: @signed_sub_r1(
; CHECK: @llvm.ssub.with.overflow.i32
entry:
  %cmp = icmp eq i32 %x, -2147483648
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %x, i32 1)
  %1 = extractvalue { i32, i1 } %0, 0
  %2 = extractvalue { i32, i1 } %0, 1
  br i1 %2, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %1, %cond.false ]
  ret i32 %cond
}

define i32 @unsigned_sub_r1(i32 %x) {
; CHECK-LABEL: @unsigned_sub_r1(
; CHECK: @llvm.usub.with.overflow.i32
entry:
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %x, i32 1)
  %1 = extractvalue { i32, i1 } %0, 0
  %2 = extractvalue { i32, i1 } %0, 1
  br i1 %2, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %1, %cond.false ]
  ret i32 %cond
}

define i32 @signed_add_rn1(i32 %x) {
; CHECK-LABEL: @signed_add_rn1(
; CHECK-NOT: @llvm.sadd.with.overflow.i32
entry:
  %cmp = icmp eq i32 %x, -2147483648
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %x, i32 -1)
  %1 = extractvalue { i32, i1 } %0, 0
  %2 = extractvalue { i32, i1 } %0, 1
  br i1 %2, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %1, %cond.false ]
  ret i32 %cond
}

define i32 @signed_sub_rn1(i32 %x) {
; CHECK-LABEL: @signed_sub_rn1(
; CHECK: @llvm.ssub.with.overflow.i32
entry:
  %cmp = icmp eq i32 %x, 2147483647
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %x, i32 -1)
  %1 = extractvalue { i32, i1 } %0, 0
  %2 = extractvalue { i32, i1 } %0, 1
  br i1 %2, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %1, %cond.false ]
  ret i32 %cond
}

declare i32 @bar(i32)

define void @unsigned_loop(i32 %i) {
; CHECK-LABEL: @unsigned_loop(
; CHECK: @llvm.usub.with.overflow.i32
entry:
  %cmp3 = icmp eq i32 %i, 0
  br i1 %cmp3, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %cont
  %i.addr.04 = phi i32 [ %2, %cont ], [ %i, %while.body.preheader ]
  %call = tail call i32 @bar(i32 %i.addr.04)
  %0 = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %i.addr.04, i32 1)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:                                             ; preds = %while.body
  tail call void @llvm.trap()
  unreachable

cont:                                             ; preds = %while.body
  %2 = extractvalue { i32, i1 } %0, 0
  %cmp = icmp eq i32 %2, 0
  br i1 %cmp, label %while.end, label %while.body

while.end:                                        ; preds = %cont, %entry
  ret void
}

define void @intrinsic_into_phi(i32 %n) {
; CHECK-LABEL: @intrinsic_into_phi(
; CHECK: @llvm.sadd.with.overflow.i32
; CHECK-NOT: @llvm.sadd.with.overflow.i32
entry:
  br label %cont

for.cond:                                         ; preds = %while.end
  %0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %.lcssa, i32 1)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:                                             ; preds = %for.cond, %while.body
  tail call void @llvm.trap()
  unreachable

cont:                                             ; preds = %entry, %for.cond
  %2 = phi { i32, i1 } [ zeroinitializer, %entry ], [ %0, %for.cond ]
  %3 = extractvalue { i32, i1 } %2, 0
  %call9 = tail call i32 @bar(i32 %3)
  %tobool10 = icmp eq i32 %call9, 0
  br i1 %tobool10, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %cont
  br label %while.body

while.cond:                                       ; preds = %while.body
  %4 = extractvalue { i32, i1 } %6, 0
  %call = tail call i32 @bar(i32 %4)
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %while.end, label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.cond
  %5 = phi i32 [ %4, %while.cond ], [ %3, %while.body.preheader ]
  %6 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %5, i32 1)
  %7 = extractvalue { i32, i1 } %6, 1
  br i1 %7, label %trap, label %while.cond

while.end:                                        ; preds = %while.cond, %cont
  %.lcssa = phi i32 [ %3, %cont ], [ %4, %while.cond ]
  %cmp = icmp slt i32 %.lcssa, %n
  br i1 %cmp, label %for.cond, label %cleanup2

cleanup2:                                         ; preds = %while.end
  ret void
}
