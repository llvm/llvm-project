; Check that bitcast in "CG Profile" related metadata nodes (in this test case,
; generated during function importing in IRMover's RAUW operations) are accepted
; by verifier.
; RUN: opt  -passes=cg-profile -module-summary %s -o %t.bc
; RUN: opt  -module-summary %p/Inputs/cg_profile.ll -o %t2.bc
; RUN: llvm-lto  -thinlto -o %t3 %t.bc %t2.bc
; RUN: opt  -passes=function-import -print-imports -summary-file %t3.thinlto.bc %t.bc \
; RUN:   -S 2>&1 | FileCheck %s

; CHECK:      !0 = !{i32 1, !"EnableSplitLTOUnit", i32 0}
; CHECK-NEXT: !1 = !{i32 5, !"CG Profile", !2}
; CHECK-NEXT: !2 = distinct !{!3}
; CHECK-NEXT: !3 = !{ptr @foo, ptr @bar, i64 2753}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; %class.A is defined differently in %p/Inputs/cg_profile.ll. This is to trigger
; bitcast.
%class.A = type { i8 }

define void @foo() !prof !2 {
  call void @bar(ptr null)
  ret void
}

declare void @bar(ptr)

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"EnableSplitLTOUnit", i32 0}
!2 = !{!"function_entry_count", i64 2753}
