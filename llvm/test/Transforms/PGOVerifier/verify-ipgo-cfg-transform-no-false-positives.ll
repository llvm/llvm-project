; REQUIRES: asserts
; RUN: llvm-profdata merge %S/Inputs/verify-ipgo-block-flow-conservation.proftext -o %t.profdata
; RUN: opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=pgo-instr-use,simplifycfg -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck %s --check-prefix=SIMPLIFY --implicit-check-not="PGOVerify# Block frequency mismatch"
; RUN: opt < %s -verify-ipgo -passes=pgo-instr-use,simplifycfg -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck %s --check-prefix=SIMPLIFY-VERIFY --implicit-check-not="PGOVerify# Block frequency mismatch"
; RUN: llvm-profdata merge %S/Inputs/verify-ipgo-block-flow-conservation.proftext -o %t.profdata
; RUN: opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=pgo-instr-use,jump-threading -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck %s --check-prefix=JTHREAD --implicit-check-not="PGOVerify# Block frequency mismatch"
; RUN: opt < %s -verify-ipgo -passes=pgo-instr-use,jump-threading -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck %s --check-prefix=JTHREAD-VERIFY --implicit-check-not="PGOVerify# Block frequency mismatch"

; Mother-patch regression for CFG-changing transforms:
; verify-ipgo should not report block-flow mismatches after simplifycfg/jump-threading
; when flow is profile-consistent.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @cfg_simplify(i32 %x) !prof !0 {
entry:
  %cmp = icmp sgt i32 %x, 0
  br i1 %cmp, label %then, label %else, !prof !1

then:
  br label %join

else:
  br label %join

join:
  %v = phi i32 [ 1, %then ], [ 1, %else ]
  br i1 true, label %ret, label %dead, !prof !4

dead:
  unreachable

ret:
  ret i32 %v
}

define i32 @cfg_thread(i1 %c) !prof !5 {
entry:
  br i1 %c, label %left, label %right, !prof !6

left:
  br label %merge

right:
  br label %merge

merge:
  %p = phi i1 [ true, %left ], [ false, %right ]
  br i1 %p, label %taken, label %nottaken, !prof !9

taken:
  ret i32 1

nottaken:
  ret i32 0
}

!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = !{!"function_entry_count", i64 1000}
!1 = !{!"branch_weights", i32 600, i32 400}
!2 = !{!"branch_weights", i32 600}
!3 = !{!"branch_weights", i32 400}
!4 = !{!"branch_weights", i32 1000, i32 0}

!5 = !{!"function_entry_count", i64 1000}
!6 = !{!"branch_weights", i32 700, i32 300}
!7 = !{!"branch_weights", i32 700}
!8 = !{!"branch_weights", i32 300}
!9 = !{!"branch_weights", i32 700, i32 300}

!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"uwtable", i32 2}
!12 = !{!"clang version 21.1.8"}

; SIMPLIFY: *** IPGO Verification After PGOInstrumentationUse ***
; SIMPLIFY: *** IPGO Verification After SimplifyCFGPass ***

; SIMPLIFY-VERIFY: *** IPGO Verification After PGOInstrumentationUse ***
; SIMPLIFY-VERIFY: *** IPGO Verification After SimplifyCFGPass ***

; JTHREAD: *** IPGO Verification After PGOInstrumentationUse ***
; JTHREAD: *** IPGO Verification After JumpThreadingPass ***

; JTHREAD-VERIFY: *** IPGO Verification After PGOInstrumentationUse ***
; JTHREAD-VERIFY: *** IPGO Verification After JumpThreadingPass ***
