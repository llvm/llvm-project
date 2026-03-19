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

define i32 @store_clustering_1(ptr nocapture %p, i32 %v) {
; NOCLUSTER: ********** MI Scheduling **********
; NOCLUSTER-LABEL: store_clustering_1:%bb.0
; NOCLUSTER: *** Final schedule for %bb.0 ***
; NOCLUSTER: SU(2):   SW %1:gpr, %0:gpr, 12 :: (store (s32) into %ir.arrayidx0)
; NOCLUSTER: SU(3):   SW %1:gpr, %0:gpr, 8 :: (store (s32) into %ir.arrayidx1)
; NOCLUSTER: SU(4):   SW %1:gpr, %0:gpr, 4 :: (store (s32) into %ir.arrayidx2)
; NOCLUSTER: SU(5):   SW %1:gpr, %0:gpr, 16 :: (store (s32) into %ir.arrayidx3)
;
; STCLUSTER: ********** MI Scheduling **********
; STCLUSTER-LABEL: store_clustering_1:%bb.0
; STCLUSTER: *** Final schedule for %bb.0 ***
; STCLUSTER: SU(4):   SW %1:gpr, %0:gpr, 4 :: (store (s32) into %ir.arrayidx2)
; STCLUSTER: SU(3):   SW %1:gpr, %0:gpr, 8 :: (store (s32) into %ir.arrayidx1)
; STCLUSTER: SU(2):   SW %1:gpr, %0:gpr, 12 :: (store (s32) into %ir.arrayidx0)
; STCLUSTER: SU(5):   SW %1:gpr, %0:gpr, 16 :: (store (s32) into %ir.arrayidx3)
;
; LDCLUSTER: ********** MI Scheduling **********
; LDCLUSTER-LABEL: store_clustering_1:%bb.0
; LDCLUSTER: *** Final schedule for %bb.0 ***
; LDCLUSTER: SU(2):   SW %1:gpr, %0:gpr, 12 :: (store (s32) into %ir.arrayidx0)
; LDCLUSTER: SU(3):   SW %1:gpr, %0:gpr, 8 :: (store (s32) into %ir.arrayidx1)
; LDCLUSTER: SU(4):   SW %1:gpr, %0:gpr, 4 :: (store (s32) into %ir.arrayidx2)
; LDCLUSTER: SU(5):   SW %1:gpr, %0:gpr, 16 :: (store (s32) into %ir.arrayidx3)
;
; DEFAULTCLUSTER: ********** MI Scheduling **********
; DEFAULTCLUSTER-LABEL: store_clustering_1:%bb.0
; DEFAULTCLUSTER: *** Final schedule for %bb.0 ***
; DEFAULTCLUSTER: SU(4):   SW %1:gpr, %0:gpr, 4 :: (store (s32) into %ir.arrayidx2)
; DEFAULTCLUSTER: SU(3):   SW %1:gpr, %0:gpr, 8 :: (store (s32) into %ir.arrayidx1)
; DEFAULTCLUSTER: SU(2):   SW %1:gpr, %0:gpr, 12 :: (store (s32) into %ir.arrayidx0)
; DEFAULTCLUSTER: SU(5):   SW %1:gpr, %0:gpr, 16 :: (store (s32) into %ir.arrayidx3)
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %p, i32 3
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i32, ptr %p, i32 2
  store i32 %v, ptr %arrayidx1
  %arrayidx2 = getelementptr inbounds i32, ptr %p, i32 1
  store i32 %v, ptr %arrayidx2
  %arrayidx3 = getelementptr inbounds i32, ptr %p, i32 4
  store i32 %v, ptr %arrayidx3
  ret i32 %v
}
