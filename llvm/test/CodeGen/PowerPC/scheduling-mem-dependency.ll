; REQUIRES: asserts
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s --check-prefix=CHECK-P9

define i64 @store_disjoint_memory(ptr nocapture %P, i64 %v) {
entry:
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_disjoint_memory:%bb.0
; CHECK:SU([[REG2:[0-9]+]]):   STD renamable $x{{[0-9]+}}, 24, renamable $x[[REG5:[0-9]+]]
; CHECK-NOT: Successors:
; CHECK-NOT:    SU([[REG3]]): Ord  Latency=0 Memory
; CHECK:SU([[REG3:[0-9]+]]):   STD renamable $x{{[0-9]+}}, 16, renamable $x[[REG5]]
; CHECK: Predecessors:
; CHECK-NOT:    SU([[REG2]]): Ord  Latency=0 Memory
  %arrayidx = getelementptr inbounds i64, ptr %P, i64 3
  store i64 %v, ptr %arrayidx
  %arrayidx1 = getelementptr inbounds i64, ptr %P, i64 2
  store i64 %v, ptr %arrayidx1
  ret i64 %v
}

; LXSD is an instruction that can be modeled.
@gd = external local_unnamed_addr global [500 x double], align 8
@gf = external local_unnamed_addr global [500 x float], align 4

define double @test_lxsd_no_barrier(double %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, double %i, double %j, double %k, double %l, double %m) {
entry:
  %0 = load double, ptr getelementptr inbounds ([500 x double], ptr @gd, i64 0, i64 10), align 8
  %1 = load double, ptr getelementptr inbounds ([500 x double], ptr @gd, i64 0, i64 17), align 8
  %2 = load double, ptr getelementptr inbounds ([500 x double], ptr @gd, i64 0, i64 87), align 8
  %3 = load double, ptr getelementptr inbounds ([500 x double], ptr @gd, i64 0, i64 97), align 8
  %4 = load double, ptr getelementptr inbounds ([500 x double], ptr @gd, i64 0, i64 77), align 8
  %add = fadd double %a, %b
  %add1 = fadd double %add, %c
  %add2 = fadd double %add1, %d
  %add3 = fadd double %add2, %e
  %add4 = fadd double %add3, %f
  %add5 = fadd double %add4, %g
  %add6 = fadd double %add5, %h
  %add7 = fadd double %add6, %i
  %add8 = fadd double %add7, %j
  %add9 = fadd double %add8, %k
  %add10 = fadd double %add9, %l
  %add11 = fadd double %add10, %m
  %add12 = fadd double %add11, %0
  %add13 = fadd double %add12, %1
  %add14 = fadd double %add13, %2
  %add15 = fadd double %add14, %3
  %add16 = fadd double %add15, %4
  ret double %add16
; CHECK-P9: ********** MI Scheduling **********
; CHECK-P9-LABEL: test_lxsd_no_barrier:%bb.0 entry
; CHECK-P9-NOT:Global memory object and new barrier chain: SU({{[0-9]+}}).
; CHECK-P9:SU({{[0-9]+}}):   renamable $vf{{[0-9]+}} = LXSD 136
; CHECK-P9:SU({{[0-9]+}}):   renamable $vf{{[0-9]+}} = LXSD 696
; CHECK-P9:SU({{[0-9]+}}):   renamable $vf{{[0-9]+}} = LXSD 776
; CHECK-P9:SU({{[0-9]+}}):   renamable $vf{{[0-9]+}} = LXSD 616
}
