; RUN: opt -codegenprepare -S -mtriple=x86_64 < %s | FileCheck %s

@exit_addr = constant ptr blockaddress(@gep_unmerging, %exit)
@op1_addr = constant ptr blockaddress(@gep_unmerging, %op1)
@op2_addr = constant ptr blockaddress(@gep_unmerging, %op2)
@op3_addr = constant ptr blockaddress(@gep_unmerging, %op3)
@dummy = global i8 0

define void @gep_unmerging(i1 %pred, ptr %p0) {
entry:
  %table = alloca [256 x ptr]
  %table_1 = getelementptr [256 x ptr], ptr %table, i64 0, i64 1
  %table_2 = getelementptr [256 x ptr], ptr %table, i64 0, i64 2
  %table_3 = getelementptr [256 x ptr], ptr %table, i64 0, i64 3
  %exit_a = load ptr, ptr @exit_addr
  %op1_a = load ptr, ptr @op1_addr
  %op2_a = load ptr, ptr @op2_addr
  %op3_a = load ptr, ptr @op3_addr
  store ptr %exit_a, ptr %table
  store ptr %op1_a, ptr %table_1
  store ptr %op2_a, ptr %table_2
  store ptr %op3_a, ptr %table_3
  br label %indirectbr

op1:
; CHECK-LABEL: op1:
; CHECK-NEXT: %p1_inc2 = getelementptr i8, ptr %p_preinc, i64 3
; CHECK-NEXT: %p1_inc1 = getelementptr i8, ptr %p_preinc, i64 2
  %p1_inc2 = getelementptr i8, ptr %p_preinc, i64 3
  %p1_inc1 = getelementptr i8, ptr %p_preinc, i64 2
  %a10 = load i8, ptr %p_postinc
  %a11 = load i8, ptr %p1_inc1
  %a12 = add i8 %a10, %a11
  store i8 %a12, ptr @dummy
  br i1 %pred, label %indirectbr, label %exit

op2:
; CHECK-LABEL: op2:
; CHECK-NEXT: %p2_inc = getelementptr i8, ptr %p_preinc, i64 2
  %p2_inc = getelementptr i8, ptr %p_preinc, i64 2
  %a2 = load i8, ptr %p_postinc
  store i8 %a2, ptr @dummy
  br i1 %pred, label %indirectbr, label %exit

op3:
  br i1 %pred, label %indirectbr, label %exit

indirectbr:
  %p_preinc = phi ptr [%p0, %entry], [%p1_inc2, %op1], [%p2_inc, %op2], [%p_postinc, %op3]
  %p_postinc = getelementptr i8, ptr %p_preinc, i64 1
  %next_op = load i8, ptr %p_preinc
  %p_zext = zext i8 %next_op to i64
  %slot = getelementptr [256 x ptr], ptr %table, i64 0, i64 %p_zext 
  %target = load ptr, ptr %slot
  indirectbr ptr %target, [label %exit, label %op1, label %op2]

exit:
  ret void
}
