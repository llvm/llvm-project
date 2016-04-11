; RUN: llc -march=mips -relocation-model=pic < %s | FileCheck %s

define void @f1(i64 %ll1, float %f, i64 %ll, i32 %i, float %f2) nounwind {
entry:
; CHECK-DAG: lw $[[R2:[0-9]+]], 80($sp)
; CHECK-DAG: lw $[[R3:[0-9]+]], 84($sp)
; CHECK-DAG: move $[[R1:[0-9]+]], $5
; CHECK-DAG: move $[[R0:[0-9]+]], $4
; CHECK-DAG: ori $6, ${{[0-9]+}}, 3855
; CHECK-DAG: ori $7, ${{[0-9]+}}, 22136
; CHECK-DAG: lw  $25, %call16(ff1)
; CHECK: jalr
  tail call void @ff1(i32 %i, i64 1085102592623924856) nounwind
; CHECK-DAG: lw $25, %call16(ff2)
; CHECK-DAG: move $4, $[[R2]]
; CHECK-DAG: move $5, $[[R3]]
; CHECK: jalr $25
  tail call void @ff2(i64 %ll, double 3.000000e+00) nounwind
  %sub = add nsw i32 %i, -1
; CHECK-DAG: lw $25, %call16(ff3)
; CHECK-DAG: sw $[[R1]], 28($sp)
; CHECK-DAG: sw $[[R0]], 24($sp)
; CHECK-DAG: move $6, $[[R2]]
; CHECK-DAG: move $7, $[[R3]]
; CHECK:     jalr $25
  tail call void @ff3(i32 %i, i64 %ll, i32 %sub, i64 %ll1) nounwind
  ret void
}

declare void @ff1(i32, i64)

declare void @ff2(i64, double)

declare void @ff3(i32, i64, i32, i64)
