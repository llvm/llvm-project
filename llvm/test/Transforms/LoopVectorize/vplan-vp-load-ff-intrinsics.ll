; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -enable-early-exit-with-ffload \
; RUN: -disable-output < %s 2>&1 | FileCheck %s

define i64 @find_with_liveout(ptr %first, i8 %value) {
; CHECK: VPlan 'Final VPlan for VF={2,4},UF={1}' {
; CHECK-NEXT: Live-in ir<1024> = vector-trip-count
; CHECK-NEXT: Live-in ir<1024> = original trip-count
; CHECK: vector.body:
; CHECK-NEXT:   EMIT-SCALAR vp<[[IV:%.+]]> = phi [ ir<0>, vector.ph ], [ vp<%index.next>, vector.body.interim ]
; CHECK-NEXT:   EMIT vp<[[REMAINDER0:%.+]]> = sub ir<1024>, vp<[[IV]]>
; CHECK-NEXT:   EMIT vp<[[COND:%.+]]> = icmp ule ir<4>, vp<[[REMAINDER0]]>
; CHECK-NEXT:   EMIT vp<[[REMAINDER:%.+]]> = select vp<[[COND]]>, ir<4>, vp<[[REMAINDER0]]>
; CHECK-NEXT:   EMIT-SCALAR vp<[[REMAINDER32:%.+]]> = trunc vp<[[REMAINDER]]> to i32
; CHECK-NEXT:   CLONE ir<%addr> = getelementptr inbounds ir<%first>, vp<[[IV]]>
; CHECK-NEXT:   WIDEN-INTRINSIC vp<[[STRUCT:%.+]]> = call llvm.vp.load.ff(ir<%addr>, ir<true>, vp<[[REMAINDER32]]>)
; CHECK-NEXT:   EMIT-SCALAR vp<[[FAULTINGLANE:%.+]]> = extract-scalar-value vp<[[STRUCT]]>, ir<1>
; CHECK-NEXT:   EMIT-SCALAR vp<[[FAULTINGLANE64:%.+]]> = zext vp<[[FAULTINGLANE]]> to i64
; CHECK-NEXT:   EMIT vp<[[ALM:%.+]]> = active lane mask ir<0>, vp<[[FAULTINGLANE64]]>, ir<1>
; CHECK-NEXT:   EMIT vp<[[DATA:%.+]]> = extract-vector-value vp<[[STRUCT]]>, ir<0>
; CHECK-NEXT:   WIDEN ir<[[EEMASK:%.+]]> = icmp eq vp<[[DATA]]>, vp<[[VALUE:%.+]]>
; CHECK-NEXT:   EMIT vp<%index.next> = add nuw vp<[[IV]]>, vp<[[FAULTINGLANE64]]>
; CHECK-NEXT:   EMIT vp<[[ALM1:%.+]]> = logical-and vp<[[ALM]]>, ir<[[EEMASK]]>
; CHECK-NEXT:   EMIT vp<[[EARLYEXIT:%.+]]> = any-of vp<[[ALM1]]>
; CHECK-NEXT:   EMIT vp<[[MAINEXIT:%.+]]> = icmp eq vp<%index.next>, ir<1024>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[EARLYEXIT]]>
; CHECK-NEXT: Successor(s): vector.early.exit, vector.body.interim
; CHECK-EMPTY:
; CHECK-NEXT:vector.body.interim:
; CHECK-NEXT:  EMIT branch-on-cond vp<[[MAINEXIT]]>
; CHECK-NEXT:Successor(s): middle.block, vector.body
; CHECK-EMPTY:
; CHECK-NEXT:middle.block:
; CHECK-NEXT:Successor(s): ir-bb<exit>
; CHECK-EMPTY:
; CHECK-NEXT:vector.early.exit:
; CHECK-NEXT:  EMIT vp<%16> = first-active-lane ir<[[EEMASK]]>
; CHECK-NEXT:  EMIT vp<%17> = add vp<%index>, vp<%16>
; CHECK-NEXT:Successor(s): ir-bb<exit>
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %addr = getelementptr inbounds i8, ptr %first, i64 %iv
  %1 = load i8, ptr %addr, align 1
  %cmp1 = icmp eq i8 %1, %value
  br i1 %cmp1, label %exit, label %for.inc

for.inc:
  %iv.next = add i64 %iv, 1
  %cmp.not = icmp eq i64 %iv.next, 1024
  br i1 %cmp.not, label %exit, label %for.body

exit:
  %retval = phi i64 [ %iv, %for.body ], [ 1024, %for.inc ]
  ret i64 %retval
}
