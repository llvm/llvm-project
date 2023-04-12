; RUN: opt < %s -simple-loop-unswitch-inject-invariant-conditions=true -passes='loop(simple-loop-unswitch<nontrivial>)' -S | FileCheck %s
; REQUIRES: asserts
; XFAIL: *

@global = external dso_local local_unnamed_addr global i64, align 8

define dso_local void @test() local_unnamed_addr #0 {
; CHECK-LABEL: test
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %call = tail call noundef ptr @widget()
  %icmp = icmp slt ptr %call, null
  br i1 %icmp, label %bb2, label %bb1

bb2:                                              ; preds = %bb1
  ret void
}

declare ptr @widget() local_unnamed_addr #0
