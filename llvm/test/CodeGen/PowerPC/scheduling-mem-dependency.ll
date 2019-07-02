; REQUIRES: asserts
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s

define i64 @store_disjoint_memory(i64* nocapture %P, i64 %v) {
entry:
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_disjoint_memory:%bb.0
; CHECK:SU(2):   STD renamable $x4, 24, renamable $x5 :: (store 8 into %ir.arrayidx)
; CHECK-NOT: Successors:
; CHECK-NOT:    SU(3): Ord  Latency=0 Memory
; CHECK:SU(3):   STD renamable $x4, 16, renamable $x5 :: (store 8 into %ir.arrayidx1)
; CHECK: Predecessors:
; CHECK-NOT:    SU(2): Ord  Latency=0 Memory
  %arrayidx = getelementptr inbounds i64, i64* %P, i64 3
  store i64 %v, i64* %arrayidx
  %arrayidx1 = getelementptr inbounds i64, i64* %P, i64 2
  store i64 %v, i64* %arrayidx1
  ret i64 %v
}
