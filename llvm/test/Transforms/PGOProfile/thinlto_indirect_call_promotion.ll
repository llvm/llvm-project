; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/thinlto_indirect_call_promotion.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; RUN: opt -passes=function-import -summary-file %t3.thinlto.bc %t.bc -o %t4.bc -print-imports 2>&1 | FileCheck %s --check-prefix=IMPORTS
; IMPORTS-DAG: Import a
; IMPORTS-DAG: Import c

; RUN: opt %t4.bc -icp-lto -passes=pgo-icall-prom -S | FileCheck %s --check-prefix=ICALL-PROM
; RUN: opt %t4.bc -icp-lto -passes=pgo-icall-prom -S -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefix=PASS-REMARK
; PASS-REMARK: Promote indirect call to a with count 1 out of 1
; PASS-REMARK: Promote indirect call to c.llvm.0 with count 1 out of 1

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = external local_unnamed_addr global void ()*, align 8
@bar = external local_unnamed_addr global void ()*, align 8

define i32 @main() local_unnamed_addr {
entry:
  %0 = load void ()*, void ()** @foo, align 8
; ICALL-PROM:   br i1 %{{[0-9]+}}, label %if.true.direct_targ, label %if.false.orig_indirect, !prof [[BRANCH_WEIGHT:![0-9]+]]
  tail call void %0(), !prof !1
  %1 = load void ()*, void ()** @bar, align 8
; ICALL-PROM:   br i1 %{{[0-9]+}}, label %if.true.direct_targ1, label %if.false.orig_indirect2, !prof [[BRANCH_WEIGHT:![0-9]+]]
  tail call void %1(), !prof !2
  ret i32 0
}

!1 = !{!"VP", i32 0, i64 1, i64 -6289574019528802036, i64 1}
!2 = !{!"VP", i32 0, i64 1, i64 591260329866125152, i64 1}

; Should not have a VP annotation on new indirect call (check before and after
; branch_weights annotation).
; ICALL-PROM-NOT: !"VP"
; ICALL-PROM: [[BRANCH_WEIGHT]] = !{!"branch_weights", i32 1, i32 0}
; ICALL-PROM-NOT: !"VP"
