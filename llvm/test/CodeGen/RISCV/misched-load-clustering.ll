; REQUIRES: asserts
; RUN: llc -mtriple=riscv32 -verify-misched -riscv-misched-load-store-clustering=false \
; RUN:     -debug-only=machine-scheduler -o - 2>&1 < %s \
; RUN:   | FileCheck -check-prefix=NOCLUSTER %s
; RUN: llc -mtriple=riscv64 -verify-misched -riscv-misched-load-store-clustering=false \
; RUN:     -debug-only=machine-scheduler -o - 2>&1 < %s \
; RUN:   | FileCheck -check-prefix=NOCLUSTER %s
; RUN: llc -mtriple=riscv32 -verify-misched \
; RUN:     -debug-only=machine-scheduler -o - 2>&1 < %s \
; RUN:   | FileCheck -check-prefix=LDCLUSTER %s
; RUN: llc -mtriple=riscv64 -verify-misched \
; RUN:     -debug-only=machine-scheduler -o - 2>&1 < %s \
; RUN:   | FileCheck -check-prefix=LDCLUSTER %s


define i32 @load_clustering_1(ptr nocapture %p) {
; NOCLUSTER: ********** MI Scheduling **********
; NOCLUSTER-LABEL: load_clustering_1:%bb.0
; NOCLUSTER: *** Final schedule for %bb.0 ***
; NOCLUSTER: SU(1): %1:gpr = LW %0:gpr, 12
; NOCLUSTER: SU(2): %2:gpr = LW %0:gpr, 8
; NOCLUSTER: SU(4): %4:gpr = LW %0:gpr, 4
; NOCLUSTER: SU(5): %6:gpr = LW %0:gpr, 16
;
; LDCLUSTER: ********** MI Scheduling **********
; LDCLUSTER-LABEL: load_clustering_1:%bb.0
; LDCLUSTER: *** Final schedule for %bb.0 ***
; LDCLUSTER: SU(4): %4:gpr = LW %0:gpr, 4
; LDCLUSTER: SU(2): %2:gpr = LW %0:gpr, 8
; LDCLUSTER: SU(1): %1:gpr = LW %0:gpr, 12
; LDCLUSTER: SU(5): %6:gpr = LW %0:gpr, 16
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %p, i32 3
  %val0 = load i32, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i32, ptr %p, i32 2
  %val1 = load i32, ptr %arrayidx1
  %tmp0 = add i32 %val0, %val1
  %arrayidx2 = getelementptr inbounds i32, ptr %p, i32 1
  %val2 = load i32, ptr %arrayidx2
  %tmp1 = add i32 %tmp0, %val2
  %arrayidx3 = getelementptr inbounds i32, ptr %p, i32 4
  %val3 = load i32, ptr %arrayidx3
  %tmp2 = add i32 %tmp1, %val3
  ret i32 %tmp2
}
