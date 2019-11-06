; REQUIRES: asserts
; RUN: llc -mtriple=riscv32 -debug-only=machine-scheduler < %s \
; RUN:   -o /dev/null 2>&1 | FileCheck %s
; RUN: llc -mtriple=riscv64 -debug-only=machine-scheduler < %s \
; RUN:   -o /dev/null 2>&1 | FileCheck %s

; This test exercises the areMemAccessesTriviallyDisjoint hook.
; Test that the two stores are disjoint memory accesses. If the corresponding
; store machine instructions don't depend on each other, the second store should
; not appear in the successors list of the first one and the first one should
; not appear on the predecessors list of the second one.
define i32 @test_disjoint(i32* %P, i32 %v) {
entry:
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: test_disjoint:%bb.0
; CHECK:SU(2):   SW %1:gpr, %0:gpr, 12 :: (store 4 into %ir.arrayidx)
; CHECK-NOT: Successors:
; CHECK:SU(3):   SW %1:gpr, %0:gpr, 8 :: (store 4 into %ir.arrayidx1)
; CHECK: Predecessors:
; CHECK-NOT:    SU(2): Ord  Latency=0 Memory
  %arrayidx = getelementptr inbounds i32, i32* %P, i32 3
  store i32 %v, i32* %arrayidx
  %arrayidx1 = getelementptr inbounds i32, i32* %P, i32 2
  store i32 %v, i32* %arrayidx1
  ret i32 %v
}
