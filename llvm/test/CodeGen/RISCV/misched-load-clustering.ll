; REQUIRES: asserts
;
; Disable all misched clustering
; RUN: llc -mtriple=riscv32 -verify-misched \
; RUN:     -mattr=+disable-misched-load-clustering,+disable-misched-store-clustering \
; RUN:     -debug-only=machine-scheduler -o - 2>&1 < %s \
; RUN:   | FileCheck -check-prefix=NOCLUSTER %s
; RUN: llc -mtriple=riscv64 -verify-misched \
; RUN:     -mattr=+disable-misched-load-clustering,+disable-misched-store-clustering \
; RUN:     -debug-only=machine-scheduler -o - 2>&1 < %s \
; RUN:   | FileCheck -check-prefix=NOCLUSTER %s
;
; ST misched clustering only
; RUN: llc -mtriple=riscv32 -verify-misched \
; RUN:     -mattr=+disable-misched-load-clustering \
; RUN:     -debug-only=machine-scheduler -o - 2>&1 < %s \
; RUN:   | FileCheck -check-prefix=STCLUSTER %s
; RUN: llc -mtriple=riscv64 -verify-misched \
; RUN:     -mattr=+disable-misched-load-clustering \
; RUN:     -debug-only=machine-scheduler -o - 2>&1 < %s \
; RUN:   | FileCheck -check-prefix=STCLUSTER %s
;
; LD misched clustering only
; RUN: llc -mtriple=riscv32 -verify-misched \
; RUN:     -mattr=+disable-misched-store-clustering \
; RUN:     -debug-only=machine-scheduler -o - 2>&1 < %s \
; RUN:   | FileCheck -check-prefix=LDCLUSTER %s
; RUN: llc -mtriple=riscv64 -verify-misched \
; RUN:     -mattr=+disable-misched-store-clustering \
; RUN:     -debug-only=machine-scheduler -o - 2>&1 < %s \
; RUN:   | FileCheck -check-prefix=LDCLUSTER %s
;
; Default misched cluster settings (i.e. both LD and ST clustering)
; RUN: llc -mtriple=riscv32 -verify-misched \
; RUN:     -debug-only=machine-scheduler -o - 2>&1 < %s \
; RUN:   | FileCheck -check-prefix=DEFAULTCLUSTER %s
; RUN: llc -mtriple=riscv64 -verify-misched \
; RUN:     -debug-only=machine-scheduler -o - 2>&1 < %s \
; RUN:   | FileCheck -check-prefix=DEFAULTCLUSTER %s

define i32 @load_clustering_1(ptr nocapture %p) {
; NOCLUSTER: ********** MI Scheduling **********
; NOCLUSTER-LABEL: load_clustering_1:%bb.0
; NOCLUSTER: *** Final schedule for %bb.0 ***
; NOCLUSTER: SU(1): %1:gpr = LW %0:gpr, 12
; NOCLUSTER: SU(2): %2:gpr = LW %0:gpr, 8
; NOCLUSTER: SU(4): %4:gpr = LW %0:gpr, 4
; NOCLUSTER: SU(5): %6:gpr = LW %0:gpr, 16
;
; STCLUSTER: ********** MI Scheduling **********
; STCLUSTER-LABEL: load_clustering_1:%bb.0
; STCLUSTER: *** Final schedule for %bb.0 ***
; STCLUSTER: SU(1): %1:gpr = LW %0:gpr, 12
; STCLUSTER: SU(2): %2:gpr = LW %0:gpr, 8
; STCLUSTER: SU(4): %4:gpr = LW %0:gpr, 4
; STCLUSTER: SU(5): %6:gpr = LW %0:gpr, 16
;
; LDCLUSTER: ********** MI Scheduling **********
; LDCLUSTER-LABEL: load_clustering_1:%bb.0
; LDCLUSTER: *** Final schedule for %bb.0 ***
; LDCLUSTER: SU(4): %4:gpr = LW %0:gpr, 4
; LDCLUSTER: SU(2): %2:gpr = LW %0:gpr, 8
; LDCLUSTER: SU(1): %1:gpr = LW %0:gpr, 12
; LDCLUSTER: SU(5): %6:gpr = LW %0:gpr, 16
;
; DEFAULTCLUSTER: ********** MI Scheduling **********
; DEFAULTCLUSTER-LABEL: load_clustering_1:%bb.0
; DEFAULTCLUSTER: *** Final schedule for %bb.0 ***
; DEFAULTCLUSTER: SU(4): %4:gpr = LW %0:gpr, 4
; DEFAULTCLUSTER: SU(2): %2:gpr = LW %0:gpr, 8
; DEFAULTCLUSTER: SU(1): %1:gpr = LW %0:gpr, 12
; DEFAULTCLUSTER: SU(5): %6:gpr = LW %0:gpr, 16
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
