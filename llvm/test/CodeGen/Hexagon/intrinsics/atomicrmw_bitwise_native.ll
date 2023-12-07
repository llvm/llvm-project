; RUN: sed -e "s/ORDER/monotonic/" -e "s/BINARY_OP/and/"  %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/acquire/"   -e "s/BINARY_OP/and/"  %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/release/"   -e "s/BINARY_OP/and/"  %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/acq_rel/"   -e "s/BINARY_OP/and/"  %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/seq_cst/"   -e "s/BINARY_OP/and/"  %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/monotonic/" -e "s/BINARY_OP/xor/"  %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/acquire/"   -e "s/BINARY_OP/xor/"  %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/release/"   -e "s/BINARY_OP/xor/"  %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/acq_rel/"   -e "s/BINARY_OP/xor/"  %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/seq_cst/"   -e "s/BINARY_OP/xor/"  %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/monotonic/" -e "s/BINARY_OP/or/"   %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/acquire/"   -e "s/BINARY_OP/or/"   %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/release/"   -e "s/BINARY_OP/or/"   %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/acq_rel/"   -e "s/BINARY_OP/or/"   %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/seq_cst/"   -e "s/BINARY_OP/or/"   %s | llc -march=hexagon | FileCheck %s

@g0 = global i32 0, align 4
@g1 = global i32 0, align 4
@g2 = global i32 0, align 4
@g3 = global i64 0, align 8
@g4 = global i64 0, align 8
@g5 = global i64 0, align 8

; CHECK-LABEL: f0:
; CHECK: // %[[BINARY_OP:[a-z_]*]]_entry
; CHECK-DAG: [[SECOND_ADDR:r[0-9]+]] = ##g1
; CHECK-DAG: [[FIRST_VALUE:r[0-9]+]] = memw(gp+#g0)

; CHECK: [[FAIL_LABEL:\.LBB.*]]:

; CHECK: [[LOCKED_READ_REG:r[0-9]+]] = memw_locked([[SECOND_ADDR]])
; CHECK: [[RESULT_REG:r[0-9]+]] = [[BINARY_OP]]([[LOCKED_READ_REG]],[[FIRST_VALUE]])
; CHECK: memw_locked([[SECOND_ADDR]],[[LOCK_PRED_REG:p[0-9]+]]) = [[RESULT_REG]]

; CHECK: if (![[LOCK_PRED_REG]]) jump{{.*}}[[FAIL_LABEL]]
; CHECK-DAG: memw(gp+#g2) = [[LOCKED_READ_REG]]
; CHECK-DAG: jumpr r31
define void @f0() {
BINARY_OP_entry:
  %v0 = load i32, ptr @g0, align 4
  %v1 = atomicrmw BINARY_OP ptr @g1, i32 %v0 ORDER
  store i32 %v1, ptr @g2, align 4
  ret void
}

; CHECK-LABEL: f1:
; CHECK-DAG: [[SECOND_ADDR:r[0-9]+]] = ##g4
; CHECK-DAG: [[FIRST_VALUE:r[:0-9]+]] = memd(gp+#g3)

; CHECK: [[FAIL_LABEL:\.LBB.*]]:

; CHECK: [[LOCKED_READ_REG:r[:0-9]+]] = memd_locked([[SECOND_ADDR]])
; CHECK: [[RESULT_REG:r[:0-9]+]] = [[BINARY_OP]]([[LOCKED_READ_REG]],[[FIRST_VALUE]])
; CHECK: memd_locked([[SECOND_ADDR]],[[LOCK_PRED_REG:p[0-9]+]]) = [[RESULT_REG]]

; CHECK: if (![[LOCK_PRED_REG]]) jump{{.*}}[[FAIL_LABEL]]
; CHECK-DAG: memd(gp+#g5) = [[LOCKED_READ_REG]]
; CHECK-DAG: jumpr r31
define void @f1() {
b0:
  %v0 = load i64, ptr @g3, align 8
  %v1 = atomicrmw BINARY_OP ptr @g4, i64 %v0 ORDER
  store i64 %v1, ptr @g5, align 8
  ret void
}
