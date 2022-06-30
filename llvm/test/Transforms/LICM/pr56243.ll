; RUN: opt -S -passes='loop-mssa(licm,loop-simplifycfg)' < %s | FileCheck %s

; REQUIRES: asserts
; XFAIL: *

; Here we end un sinking a user of token down from the loop, therefore breaching LCSSA form.
; Then, LoopSimplifyCFG expcets that LCSSA form is maintained, and remains unaware that
; it may be penetrated by tokens. As result, it may end up breaking dominance between def and
; use by introducing fake temporary edges.

define i8 addrspace(1)* @test_gc_relocate() gc "statepoint-example" {
; CHECK-LABEL: @test_gc_relocate
  br label %bb1

bb1:                                              ; preds = %bb45, %0
  switch i32 undef, label %bb43 [
    i32 1, label %bb18
  ]

bb18:                                             ; preds = %bb1
  switch i32 undef, label %bb43 [
    i32 0, label %bb28
  ]

bb28:                                             ; preds = %bb18
  %tmp34 = call token (i64, i32, i8 addrspace(1)* (i64, i32, i32, i32)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p1i8i64i32i32i32f(i64 2882400000, i32 0, i8 addrspace(1)* (i64, i32, i32, i32)* nonnull elementtype(i8 addrspace(1)* (i64, i32, i32, i32)) @barney.4, i32 4, i32 0, i64 undef, i32 5, i32 5, i32 undef, i32 0, i32 0) [ "deopt"(), "gc-live"(i8 addrspace(1)* undef) ]
  %tmp35 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %tmp34, i32 0, i32 0) ; (undef, undef)
  br i1 false, label %bb57, label %bb36

bb36:                                             ; preds = %bb28
  switch i32 undef, label %bb43 [
    i32 1, label %bb39
  ]

bb39:                                             ; preds = %bb36
  switch i32 undef, label %bb43 [
    i32 1, label %bb45
  ]

bb43:                                             ; preds = %bb39, %bb36, %bb18, %bb1
  unreachable

bb45:                                             ; preds = %bb39
  br label %bb1

bb57:                                             ; preds = %bb28
  ret i8 addrspace(1)* %tmp35
}

declare i8 addrspace(1)* @barney.4(i64, i32, i32, i32)

declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32 immarg, i32 immarg) #0

declare token @llvm.experimental.gc.statepoint.p0f_p1i8i64i32i32i32f(i64 immarg, i32 immarg, i8 addrspace(1)* (i64, i32, i32, i32)*, i32 immarg, i32 immarg, ...)

attributes #0 = { nounwind readnone }
