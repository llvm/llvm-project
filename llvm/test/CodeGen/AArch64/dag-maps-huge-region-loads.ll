; REQUIRES: asserts
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -debug-only=machine-scheduler -dag-maps-huge-region=2 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -debug-only=machine-scheduler -dag-maps-huge-region=2 -dag-maps-huge-region-store-barrier=false 2>&1 | FileCheck %s --check-prefix=LOADBARRIER
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -debug-only=machine-scheduler -dag-maps-huge-region=2 -dag-maps-huge-region-hard-limit=3 2>&1 | FileCheck %s --check-prefix=LOADBARRIER
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -debug-only=machine-scheduler -dag-maps-huge-region=4 2>&1 | FileCheck %s --check-prefix=BUDGET

; A store-free run of loads
define double @loads_only(ptr %p) {
entry:
  %p1 = getelementptr double, ptr %p, i64 10
  %p2 = getelementptr double, ptr %p, i64 20
  %p3 = getelementptr double, ptr %p, i64 30
  %a = load double, ptr %p
  %b = load double, ptr %p1
  %c = load double, ptr %p2
  %d = load double, ptr %p3
  %s0 = fadd double %a, %b
  %s1 = fadd double %c, %d
  %s = fmul double %s0, %s1
  ret double %s
}

; CHECK-LABEL: loads_only:%bb.0 entry
; CHECK-NOT:   Barrier
; CHECK:       SU({{[0-9]+}}): {{.*}} = LDRDui {{.*}} :: (load (s64) from %ir.p3)
; CHECK-NOT:   Barrier

; LOADBARRIER-LABEL: loads_only:%bb.0 entry
; LOADBARRIER:       Ord  Latency={{[0-9]+}} Barrier
; LOADBARRIER-LABEL: store_then_loads:%bb.0 entry

; The DAG is built bottom-up, so the store above the loads is reached after the
; limit; it still becomes the barrier.
define double @store_then_loads(ptr %p, ptr %q, double %v) {
entry:
  store double %v, ptr %q
  %p1 = getelementptr double, ptr %p, i64 10
  %p2 = getelementptr double, ptr %p, i64 20
  %a = load double, ptr %p
  %b = load double, ptr %p1
  %c = load double, ptr %p2
  %s0 = fadd double %a, %b
  %s = fadd double %s0, %c
  ret double %s
}

; CHECK-LABEL: store_then_loads:%bb.0 entry
; CHECK:       Creating barrier chain and clearing maps.
; CHECK:       SU({{[0-9]+}}): STRDui {{.*}} :: (store (s64) into %ir.q)
; CHECK:       Successors:
; CHECK-DAG:     SU({{[0-9]+}}): Ord  Latency={{[0-9]+}} Barrier
; CHECK-DAG:     SU({{[0-9]+}}): Ord  Latency={{[0-9]+}} Barrier
; CHECK-DAG:     SU({{[0-9]+}}): Ord  Latency={{[0-9]+}} Barrier


; One store below five loads: at most 1 * 6 < 8, no barrier.
define void @loads_then_one_store(ptr %p, ptr %q) {
entry:
  %p1 = getelementptr double, ptr %p, i64 10
  %p2 = getelementptr double, ptr %p, i64 20
  %p3 = getelementptr double, ptr %p, i64 30
  %p4 = getelementptr double, ptr %p, i64 40
  %a = load double, ptr %p
  %b = load double, ptr %p1
  %c = load double, ptr %p2
  %d = load double, ptr %p3
  %e = load double, ptr %p4
  %s0 = fadd double %a, %b
  %s1 = fadd double %c, %d
  %s2 = fadd double %s0, %s1
  %s = fadd double %s2, %e
  store double %s, ptr %q
  ret void
}

; BUDGET-LABEL: loads_then_one_store:%bb.0 entry
; BUDGET-NOT:   Barrier
; BUDGET-LABEL: loads_then_stores:%bb.0 entry

; Three stores below five loads:
; 3 * 4 >= 8, so it becomes the barrier.
define void @loads_then_stores(ptr %p, ptr %q) {
entry:
  %p1 = getelementptr double, ptr %p, i64 10
  %p2 = getelementptr double, ptr %p, i64 20
  %p3 = getelementptr double, ptr %p, i64 30
  %p4 = getelementptr double, ptr %p, i64 40
  %a = load double, ptr %p
  %b = load double, ptr %p1
  %c = load double, ptr %p2
  %d = load double, ptr %p3
  %e = load double, ptr %p4
  %s0 = fadd double %a, %b
  %s1 = fadd double %c, %d
  %s2 = fadd double %s0, %s1
  %s = fadd double %s2, %e
  %q1 = getelementptr double, ptr %q, i64 1
  %q2 = getelementptr double, ptr %q, i64 2
  store double %s, ptr %q
  store double %s2, ptr %q1
  store double %s1, ptr %q2
  ret void
}

; BUDGET:       Creating barrier chain and clearing maps.
; BUDGET:       SU({{[0-9]+}}): {{.*}} = LDRDui {{.*}} :: (load (s64)
; BUDGET:       Successors:
; BUDGET:         SU({{[0-9]+}}): Ord  Latency={{[0-9]+}} Barrier
