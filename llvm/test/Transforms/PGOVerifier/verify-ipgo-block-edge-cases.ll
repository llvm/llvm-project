; REQUIRES: asserts
; RUN: llvm-profdata merge %S/Inputs/verify-ipgo-block-edge-cases.proftext -o %t.profdata
; RUN: opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -verify-ipgo -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck --check-prefix=VERIFY %s

; Mother-patch edge cases for block-frequency validation.
; Ensures verifier skips exit/unreachable-only blocks and reports real mismatches.

define i32 @multiple_returns(i32 %x) !prof !5 {
entry:
  %cmp1 = icmp eq i32 %x, 0
  br i1 %cmp1, label %return.zero, label %check.positive, !prof !6

check.positive:
  %cmp2 = icmp sgt i32 %x, 0
  br i1 %cmp2, label %return.positive, label %return.negative, !prof !7

return.zero:
  ret i32 0

return.positive:
  ret i32 1

return.negative:
  ret i32 -1
}

define i32 @with_unreachable(i32 %x) !prof !8 {
entry:
  %cmp = icmp sgt i32 %x, 100
  br i1 %cmp, label %normal.path, label %also.normal, !prof !9

normal.path:
  ret i32 1

also.normal:
  ret i32 2

dead.block:
  %mul = mul i32 %x, 2
  br label %more.dead

more.dead:
  unreachable
}

define i32 @nested_correct(i32 %a, i32 %b, i32 %c) !prof !10 {
entry:
  %cmp1 = icmp sgt i32 %a, 0
  br i1 %cmp1, label %outer.then, label %outer.else, !prof !11

outer.then:
  %cmp2 = icmp sgt i32 %b, 0
  br i1 %cmp2, label %inner.then, label %inner.else, !prof !12

inner.then:
  %add1 = add i32 %a, %b
  br label %join.inner

inner.else:
  %sub1 = sub i32 %a, %b
  br label %join.inner

join.inner:
  %result1 = phi i32 [ %add1, %inner.then ], [ %sub1, %inner.else ]
  br label %join.outer

outer.else:
  %mul = mul i32 %a, %c
  br label %join.outer

join.outer:
  %final = phi i32 [ %result1, %join.inner ], [ %mul, %outer.else ]
  ret i32 %final
}

define i32 @nested_incorrect(i32 %a, i32 %b, i32 %c) !prof !13 {
entry:
  %cmp1 = icmp sgt i32 %a, 0
  br i1 %cmp1, label %outer.then, label %outer.else, !prof !14

outer.then:
  %cmp2 = icmp sgt i32 %b, 0
  br i1 %cmp2, label %inner.then, label %inner.else, !prof !15

inner.then:
  %add1 = add i32 %a, %b
  br label %join.inner

inner.else:
  %sub1 = sub i32 %a, %b
  br label %join.inner

join.inner:
  %result1 = phi i32 [ %add1, %inner.then ], [ %sub1, %inner.else ]
  br label %join.outer

outer.else:
  %mul = mul i32 %a, %c
  br label %join.outer

join.outer:
  %final = phi i32 [ %result1, %join.inner ], [ %mul, %outer.else ]
  ret i32 %final
}

define i32 @switch_all_returns(i32 %x) !prof !20 {
entry:
  switch i32 %x, label %default [
    i32 1, label %case1
    i32 2, label %case2
    i32 3, label %case3
  ], !prof !21

case1:
  ret i32 10

case2:
  ret i32 20

case3:
  ret i32 30

default:
  ret i32 0
}

define i32 @with_assertion(i32 %x) !prof !22 {
entry:
  %valid = icmp sge i32 %x, 0
  br i1 %valid, label %normal, label %error, !prof !23

normal:
  %result = mul i32 %x, 2
  ret i32 %result

error:
  call void @abort() noreturn
  unreachable
}

declare void @abort() noreturn

; Function with manually set inconsistent branch weights, intentionally not listed in
; the proftext so pgo-instr-use will leave its metadata unchanged.
; outer.then: incoming=700 (from entry branch weight), outgoing=300+300=600 → mismatch.
define i32 @nested_inconsistent_manual(i32 %a, i32 %b) !prof !24 {
entry:
  %cmp1 = icmp sgt i32 %a, 0
  br i1 %cmp1, label %outer.then, label %outer.else, !prof !25

outer.then:
  %cmp2 = icmp sgt i32 %b, 0
  br i1 %cmp2, label %inner.then, label %inner.else, !prof !26

inner.then:
  ret i32 1

inner.else:
  ret i32 0

outer.else:
  ret i32 -1
}

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 21.1.8"}

!5 = !{!"function_entry_count", i64 1000}
!6 = !{!"branch_weights", i32 200, i32 800}
!7 = !{!"branch_weights", i32 500, i32 300}

!8 = !{!"function_entry_count", i64 1000}
!9 = !{!"branch_weights", i32 600, i32 400}

!10 = !{!"function_entry_count", i64 1000}
!11 = !{!"branch_weights", i32 700, i32 300}
!12 = !{!"branch_weights", i32 400, i32 300}

!13 = !{!"function_entry_count", i64 1000}
!14 = !{!"branch_weights", i32 700, i32 300}
!15 = !{!"branch_weights", i32 300, i32 400}

!20 = !{!"function_entry_count", i64 1000}
!21 = !{!"branch_weights", i32 100, i32 200, i32 300, i32 400}

!22 = !{!"function_entry_count", i64 1000}
!23 = !{!"branch_weights", i32 999, i32 1}

!24 = !{!"function_entry_count", i64 1000}
!25 = !{!"branch_weights", i32 700, i32 300}
!26 = !{!"branch_weights", i32 300, i32 300}

; CHECK: *** IPGO Verification After PGOInstrumentationUse ***
; CHECK: PGOVerify cache invalidated
; CHECK: PGOVerify# Block frequency mismatch in function nested_inconsistent_manual, block outer.then: Incoming=700: Outgoing=600

; VERIFY: *** IPGO Verification After PGOInstrumentationUse ***
; VERIFY: PGOVerify# Block frequency mismatch in function nested_inconsistent_manual, block outer.then: Incoming=700: Outgoing=600
