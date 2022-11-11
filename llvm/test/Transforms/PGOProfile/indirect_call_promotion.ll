; RUN: opt < %s -passes=pgo-icall-prom -S -icp-total-percent-threshold=50 | FileCheck %s --check-prefix=ICALL-PROM
; RUN: opt < %s -passes=pgo-icall-prom -S -pass-remarks=pgo-icall-prom -icp-remaining-percent-threshold=0 -icp-total-percent-threshold=0 -icp-max-prom=4 2>&1 | FileCheck %s --check-prefix=PASS-REMARK
; RUN: opt < %s -passes=pgo-icall-prom -S -pass-remarks=pgo-icall-prom -icp-remaining-percent-threshold=0 -icp-total-percent-threshold=20 -icp-max-prom=4 2>&1 | FileCheck %s --check-prefix=PASS2-REMARK

; PASS-REMARK: remark: <unknown>:0:0: Promote indirect call to func4 with count 1030 out of 1600
; PASS-REMARK: remark: <unknown>:0:0: Promote indirect call to func2 with count 410 out of 570
; PASS-REMARK: remark: <unknown>:0:0: Promote indirect call to func3 with count 150 out of 160
; PASS-REMARK: remark: <unknown>:0:0: Promote indirect call to func1 with count 10 out of 10

; PASS2-REMARK: remark: <unknown>:0:0: Promote indirect call to func4 with count 1030 out of 1600
; PASS2-REMARK: remark: <unknown>:0:0: Promote indirect call to func2 with count 410 out of 570
; PASS2-REMARK-NOT: remark: <unknown>:0:0: Promote indirect call to func3
; PASS2-REMARK-NOT: remark: <unknown>:0:0: Promote indirect call to func1

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = common global ptr null, align 8

define i32 @func1() {
entry:
  ret i32 0
}

define i32 @func2() {
entry:
  ret i32 1
}

define i32 @func3() {
entry:
  ret i32 2
}

define i32 @func4() {
entry:
  ret i32 3
}

define i32 @bar() {
entry:
  %tmp = load ptr, ptr @foo, align 8
; ICALL-PROM:   [[CMP:%[0-9]+]] = icmp eq ptr %tmp, @func4
; ICALL-PROM:   br i1 [[CMP]], label %if.true.direct_targ, label %if.false.orig_indirect, !prof [[BRANCH_WEIGHT:![0-9]+]]
; ICALL-PROM: if.true.direct_targ:
; ICALL-PROM:   [[DIRCALL_RET:%[0-9]+]] = call i32 @func4()
; ICALL-PROM-SAMPLEPGO: call i32 @func4(), !prof [[CALL_METADATA:![0-9]+]]
; ICALL-PROM:   br label %if.end.icp
  %call = call i32 %tmp(), !prof !1
; ICALL-PROM: if.false.orig_indirect:
; ICALL-PROM:   %call = call i32 %tmp(), !prof [[NEW_VP_METADATA:![0-9]+]]
  ret i32 %call
; ICALL-PROM: if.end.icp:
; ICALL-PROM:   [[PHI_RET:%[0-9]+]] = phi i32 [ %call, %if.false.orig_indirect ], [ [[DIRCALL_RET]], %if.true.direct_targ ]
; ICALL-PROM:   ret i32 [[PHI_RET]]
}

!1 = !{!"VP", i32 0, i64 1600, i64 7651369219802541373, i64 1030, i64 -4377547752858689819, i64 410, i64 -6929281286627296573, i64 150, i64 -2545542355363006406, i64 10}

; ICALL-PROM: [[BRANCH_WEIGHT]] = !{!"branch_weights", i32 1030, i32 570}
; ICALL-PROM: [[NEW_VP_METADATA]] = !{!"VP", i32 0, i64 570, i64 -4377547752858689819, i64 410}
; ICALL-PROM-SAMPLEPGO: [[CALL_METADATA]] = !{!"branch_weights", i32 1030}
