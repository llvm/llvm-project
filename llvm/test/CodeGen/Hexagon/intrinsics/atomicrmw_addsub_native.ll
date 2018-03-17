; RUN: sed -e "s/ORDER/monotonic/" -e "s/BINARY_OP/add/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/acquire/"   -e "s/BINARY_OP/add/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/release/"   -e "s/BINARY_OP/add/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/acq_rel/"   -e "s/BINARY_OP/add/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/seq_cst/"   -e "s/BINARY_OP/add/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/monotonic/" -e "s/BINARY_OP/sub/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/acquire/"   -e "s/BINARY_OP/sub/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/release/"   -e "s/BINARY_OP/sub/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/acq_rel/"   -e "s/BINARY_OP/sub/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/seq_cst/"   -e "s/BINARY_OP/sub/" %s | llc -march=hexagon | FileCheck %s

%struct.Obj = type { [100 x i32] }

@i32First   = global i32 0, align 4
@i32Second  = global i32 0, align 4
@i32Result  = global i32 0, align 4
@i64First   = global i64 0, align 8
@i64Second  = global i64 0, align 8
@i64Result  = global i64 0, align 8
@ptrFirst   = global %struct.Obj* null, align 4
@ptrSecond  = global %struct.Obj* null, align 4
@ptrResult  = global %struct.Obj* null, align 4

define void @atomicrmw_op_i32() #0 {
BINARY_OP_entry:
  %i32First = load i32, i32* @i32First, align 4
  %i32Result = atomicrmw BINARY_OP i32* @i32Second, i32 %i32First ORDER
  store i32 %i32Result, i32* @i32Result, align 4
  ret void
}
; CHECK-LABEL: atomicrmw_op_i32:
; CHECK: // %[[BINARY_OP:[a-z_]*]]_entry
; CHECK-DAG: [[SECOND_ADDR:r[0-9]+]] = ##i32Second
; CHECK-DAG: [[FIRST_VALUE:r[0-9]+]] = memw(gp+#i32First)

; CHECK: [[FAIL_LABEL:\.LBB.*]]:

; CHECK: [[LOCKED_READ_REG:r[0-9]+]] = memw_locked([[SECOND_ADDR]])
; CHECK: [[RESULT_REG:r[0-9]+]] = [[BINARY_OP]]([[LOCKED_READ_REG]],[[FIRST_VALUE]])
; CHECK: memw_locked([[SECOND_ADDR]],[[LOCK_PRED_REG:p[0-9]+]]) = [[RESULT_REG]]

; CHECK: cmp.eq{{.*}}jump{{.*}}[[FAIL_LABEL]]
; CHECK-DAG: memw(gp+#i32Result) = [[LOCKED_READ_REG]]
; CHECK-DAG: jumpr r31

define void @atomicrmw_op_i64() #0 {
entry:
  %i64First = load i64, i64* @i64First, align 8
  %i64Result = atomicrmw BINARY_OP i64* @i64Second, i64 %i64First ORDER
  store i64 %i64Result, i64* @i64Result, align 8
  ret void
}
; CHECK-LABEL: atomicrmw_op_i64:
; CHECK-DAG: [[SECOND_ADDR:r[0-9]+]] = ##i64Second
; CHECK-DAG: [[FIRST_VALUE:r[0-9]+:[0-9]+]] = memd(gp+#i64First)

; CHECK: [[FAIL_LABEL:\.LBB.*]]:

; CHECK-DAG: [[LOCKED_READ_REG:r[0-9]+:[0-9]+]] = memd_locked([[SECOND_ADDR]])
; CHECK: [[RESULT_REG:r[0-9]+:[0-9]+]] = [[BINARY_OP]]([[LOCKED_READ_REG]],[[FIRST_VALUE]])
; CHECK: memd_locked([[SECOND_ADDR]],[[LOCK_PRED_REG:p[0-9]+]]) = [[RESULT_REG]]

; CHECK: cmp.eq{{.*}}jump{{.*}}[[FAIL_LABEL]]
; CHECK-DAG: memd(gp+#i64Result) = [[LOCKED_READ_REG]]
; CHECK-DAG: jumpr r31

define void @atomicrmw_op_ptr() #0 {
entry:
  %ptrFirst = load i32, i32* bitcast (%struct.Obj** @ptrFirst to i32*), align 4
  %ptrResult = atomicrmw BINARY_OP i32* bitcast (%struct.Obj** @ptrSecond to i32*), i32 %ptrFirst ORDER
  store i32 %ptrResult, i32* bitcast (%struct.Obj** @ptrResult to i32*), align 4
  ret void
}
; CHECK-LABEL: atomicrmw_op_ptr:
; CHECK-DAG: [[SECOND_ADDR:r[0-9]+]] = ##ptrSecond
; CHECK-DAG: [[FIRST_VALUE:r[0-9]+]] = memw(gp+#ptrFirst)

; CHECK: [[FAIL_LABEL:\.LBB.*]]:

; CHECK-DAG: [[LOCKED_READ_REG:r[0-9]+]] = memw_locked([[SECOND_ADDR]])
; CHECK: [[RESULT_REG:r[0-9]+]] = [[BINARY_OP]]([[LOCKED_READ_REG]],[[FIRST_VALUE]])
; CHECK: memw_locked([[SECOND_ADDR]],[[LOCK_PRED_REG:p[0-9]+]]) = [[RESULT_REG]]

; CHECK: cmp.eq{{.*}}jump{{.*}}[[FAIL_LABEL]]
; CHECK-DAG: memw(gp+#ptrResult) = [[LOCKED_READ_REG]]
; CHECK-DAG: jumpr r31

