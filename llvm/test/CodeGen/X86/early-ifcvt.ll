; RUN: llc < %s -x86-early-ifcvt -stress-early-ifcvt | FileCheck %s
target triple = "x86_64-apple-macosx10.8.0"

; CHECK: mm2
define i32 @mm2(i32* nocapture %p, i32 %n) nounwind uwtable readonly ssp {
entry:
  br label %do.body

; CHECK: do.body
; Loop body has no branches before the backedge.
; CHECK-NOT: LBB
do.body:
  %max.0 = phi i32 [ 0, %entry ], [ %max.1, %do.cond ]
  %min.0 = phi i32 [ 0, %entry ], [ %min.1, %do.cond ]
  %n.addr.0 = phi i32 [ %n, %entry ], [ %dec, %do.cond ]
  %p.addr.0 = phi i32* [ %p, %entry ], [ %incdec.ptr, %do.cond ]
  %incdec.ptr = getelementptr inbounds i32* %p.addr.0, i64 1
  %0 = load i32* %p.addr.0, align 4
  %cmp = icmp sgt i32 %0, %max.0
  br i1 %cmp, label %do.cond, label %if.else

if.else:
  %cmp1 = icmp slt i32 %0, %min.0
  %.min.0 = select i1 %cmp1, i32 %0, i32 %min.0
  br label %do.cond

do.cond:
  %max.1 = phi i32 [ %0, %do.body ], [ %max.0, %if.else ]
  %min.1 = phi i32 [ %min.0, %do.body ], [ %.min.0, %if.else ]
; CHECK: decl %esi
; CHECK: jne LBB
  %dec = add i32 %n.addr.0, -1
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %do.end, label %do.body

do.end:
  %sub = sub nsw i32 %max.1, %min.1
  ret i32 %sub
}

; CHECK: multipreds
; Deal with alternative tail predecessors
; CHECK-NOT: LBB
; CHECK: cmov
; CHECK-NOT: LBB
; CHECK: cmov
; CHECK-NOT: LBB
; CHECK: fprintf

define void @multipreds(i32 %sw) nounwind uwtable ssp {
entry:
  switch i32 %sw, label %if.then29 [
    i32 0, label %if.then37
    i32 127, label %if.end41
  ]

if.then29:
  br label %if.end41

if.then37:
  br label %if.end41

if.end41:
  %exit_status.0 = phi i32 [ 2, %if.then29 ], [ 0, %if.then37 ], [ 66, %entry ]
  call void (...)* @fprintf(i32 %exit_status.0) nounwind
  unreachable
}

declare void @fprintf(...) nounwind

; CHECK: BZ2_decompress
; This test case contains irreducible control flow, so MachineLoopInfo doesn't
; recognize the cycle in the CFG. This would confuse MachineTraceMetrics.
define void @BZ2_decompress(i8* %s) nounwind ssp {
entry:
  switch i32 undef, label %sw.default [
    i32 39, label %if.end.sw.bb2050_crit_edge
    i32 36, label %sw.bb1788
    i32 37, label %if.end.sw.bb1855_crit_edge
    i32 40, label %sw.bb2409
    i32 38, label %sw.bb1983
    i32 44, label %if.end.sw.bb3058_crit_edge
  ]

if.end.sw.bb3058_crit_edge:                       ; preds = %entry
  br label %save_state_and_return

if.end.sw.bb1855_crit_edge:                       ; preds = %entry
  br label %save_state_and_return

if.end.sw.bb2050_crit_edge:                       ; preds = %entry
  br label %sw.bb2050

sw.bb1788:                                        ; preds = %entry
  br label %save_state_and_return

sw.bb1983:                                        ; preds = %entry
  br i1 undef, label %save_state_and_return, label %if.then1990

if.then1990:                                      ; preds = %sw.bb1983
  br label %while.body2038

while.body2038:                                   ; preds = %sw.bb2050, %if.then1990
  %groupPos.8 = phi i32 [ 0, %if.then1990 ], [ %groupPos.9, %sw.bb2050 ]
  br i1 undef, label %save_state_and_return, label %if.end2042

if.end2042:                                       ; preds = %while.body2038
  br i1 undef, label %if.end2048, label %while.end2104

if.end2048:                                       ; preds = %if.end2042
  %bsLive2054.pre = getelementptr inbounds i8* %s, i32 8
  br label %sw.bb2050

sw.bb2050:                                        ; preds = %if.end2048, %if.end.sw.bb2050_crit_edge
  %groupPos.9 = phi i32 [ 0, %if.end.sw.bb2050_crit_edge ], [ %groupPos.8, %if.end2048 ]
  %and2064 = and i32 undef, 1
  br label %while.body2038

while.end2104:                                    ; preds = %if.end2042
  br i1 undef, label %save_state_and_return, label %if.end2117

if.end2117:                                       ; preds = %while.end2104
  br i1 undef, label %while.body2161.lr.ph, label %while.body2145.lr.ph

while.body2145.lr.ph:                             ; preds = %if.end2117
  br label %save_state_and_return

while.body2161.lr.ph:                             ; preds = %if.end2117
  br label %save_state_and_return

sw.bb2409:                                        ; preds = %entry
  br label %save_state_and_return

sw.default:                                       ; preds = %entry
  call void @BZ2_bz__AssertH__fail() nounwind
  br label %save_state_and_return

save_state_and_return:
  %groupPos.14 = phi i32 [ 0, %sw.default ], [ %groupPos.8, %while.body2038 ], [ %groupPos.8, %while.end2104 ], [ 0, %if.end.sw.bb3058_crit_edge ], [ 0, %if.end.sw.bb1855_crit_edge ], [ %groupPos.8, %while.body2161.lr.ph ], [ %groupPos.8, %while.body2145.lr.ph ], [ 0, %sw.bb2409 ], [ 0, %sw.bb1788 ], [ 0, %sw.bb1983 ]
  store i32 %groupPos.14, i32* undef, align 4
  ret void
}

declare void @BZ2_bz__AssertH__fail()
