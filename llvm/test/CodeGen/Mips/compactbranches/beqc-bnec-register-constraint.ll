; RUN: llc -march=mips -mcpu=mips32r6 -O1 -start-after=dwarfehprepare < %s | FileCheck %s
; RUN: llc -march=mips64 -mcpu=mips64r6 -O1 -start-after=dwarfehprepare < %s | FileCheck %s


; beqc/bnec have the constraint that $rs < $rt && $rs != 0 && $rt != 0
; Cases where $rs == 0 and $rt != 0 should be transformed into beqzc/bnezc.
; Cases where $rs > $rt can have the operands swapped as ==,!= are commutative.

; Cases where beq & bne where $rs == $rt have to inhibited from being turned
; into compact branches but arguably should not occur. This test covers the
; $rs == $rt case.

; Starting from dwarf exception handling preparation skips optimizations that
; may simplify out the crucical bnec $4, $4 instruction.

define internal void @_ZL14TestRemoveLastv(ptr %alist.sroa.0.4) {
; CHECK-LABEL: _ZL14TestRemoveLastv:
entry:
  %ascevgep = getelementptr i32, ptr %alist.sroa.0.4, i64 99
  br label %do.body121

for.cond117:
  %alsr.iv.next = add nsw i32 %alsr.iv, -1
  %ascevgep340 = getelementptr i32, ptr %alsr.iv339, i64 -1
  %acmp118 = icmp sgt i32 %alsr.iv.next, 0
  br i1 %acmp118, label %do.body121, label %if.then143

do.body121:
  %alsr.iv339 = phi ptr [ %ascevgep, %entry ], [ %ascevgep340, %for.cond117 ]
  %alsr.iv = phi i32 [ 100, %entry ], [ %alsr.iv.next, %for.cond117 ]
  %a9 = add i32 %alsr.iv, -1
  %alnot124 = icmp eq i32 %alsr.iv, %alsr.iv
  br i1 %alnot124, label %do.body134, label %if.then143, !prof !11

do.body134:
  %a10 = add i32 %alsr.iv, -1
  %a11 = load i32, ptr %alsr.iv339, align 4, !tbaa !5
; CHECK-NOT: bnec $[[R0:[0-9]+]], $[[R0]]
; CHECK-NOT: beqc $[[R1:[0-9]+]], $[[R1]]
  %alnot137 = icmp eq i32 %a9, %a11
  br i1 %alnot137, label %do.end146, label %if.then143, !prof !11

if.then143:
 ret void
 unreachable

do.end146:
  %alnot151 = icmp eq i32 %a9, %a10
  br i1 %alnot151, label %for.cond117, label %if.then143, !prof !11

}

define internal void @_ZL14TestRemoveLastv64(ptr %alist.sroa.0.4) {
; CHECK-LABEL: _ZL14TestRemoveLastv64:
entry:
  %ascevgep = getelementptr i64, ptr %alist.sroa.0.4, i64 99
  br label %do.body121

for.cond117:
  %alsr.iv.next = add nsw i64 %alsr.iv, -1
  %ascevgep340 = getelementptr i64, ptr %alsr.iv339, i64 -1
  %acmp118 = icmp sgt i64 %alsr.iv.next, 0
  br i1 %acmp118, label %do.body121, label %if.then143

do.body121:
  %alsr.iv339 = phi ptr [ %ascevgep, %entry ], [ %ascevgep340, %for.cond117 ]
  %alsr.iv = phi i64 [ 100, %entry ], [ %alsr.iv.next, %for.cond117 ]
  %a9 = add i64 %alsr.iv, -1
  %alnot124 = icmp eq i64 %alsr.iv, %alsr.iv
  br i1 %alnot124, label %do.body134, label %if.then143, !prof !11

do.body134:
  %a10 = add i64 %alsr.iv, -1
  %a11 = load i64, ptr %alsr.iv339, align 4, !tbaa !5
; CHECK-NOT: bnec $[[R0:[0-9]+]], $[[R0]]
; CHECK-NOT: beqc $[[R1:[0-9]+]], $[[R1]]
  %alnot137 = icmp eq i64 %a9, %a11
  br i1 %alnot137, label %do.end146, label %if.then143, !prof !11

if.then143:
 ret void
 unreachable

do.end146:
  %alnot151 = icmp eq i64 %a9, %a10
  br i1 %alnot151, label %for.cond117, label %if.then143, !prof !11

}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !3, i64 0}
!11 = !{!"branch_weights", i32 2000, i32 1}
!12 = !{!"branch_weights", i32 -388717296, i32 7818360}

