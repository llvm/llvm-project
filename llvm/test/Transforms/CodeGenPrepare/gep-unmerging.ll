; RUN: opt -codegenprepare -S < %s | FileCheck %s

@exit_addr = constant i8* blockaddress(@gep_unmerging, %exit)
@op1_addr = constant i8* blockaddress(@gep_unmerging, %op1)
@op2_addr = constant i8* blockaddress(@gep_unmerging, %op2)
@op3_addr = constant i8* blockaddress(@gep_unmerging, %op3)
@dummy = global i8 0

define void @gep_unmerging(i1 %pred, i8* %p0) {
entry:
  %table = alloca [256 x i8*]
  %table_0 = getelementptr [256 x i8*], [256 x i8*]* %table, i64 0, i64 0
  %table_1 = getelementptr [256 x i8*], [256 x i8*]* %table, i64 0, i64 1
  %table_2 = getelementptr [256 x i8*], [256 x i8*]* %table, i64 0, i64 2
  %table_3 = getelementptr [256 x i8*], [256 x i8*]* %table, i64 0, i64 3
  %exit_a = load i8*, i8** @exit_addr
  %op1_a = load i8*, i8** @op1_addr
  %op2_a = load i8*, i8** @op2_addr
  %op3_a = load i8*, i8** @op3_addr
  store i8* %exit_a, i8** %table_0
  store i8* %op1_a, i8** %table_1
  store i8* %op2_a, i8** %table_2
  store i8* %op3_a, i8** %table_3
  br label %indirectbr

op1:
; CHECK-LABEL: op1:
; CHECK-NEXT: %p1_inc2 = getelementptr i8, i8* %p_postinc, i64 2
; CHECK-NEXT: %p1_inc1 = getelementptr i8, i8* %p_postinc, i64 1
  %p1_inc2 = getelementptr i8, i8* %p_preinc, i64 3
  %p1_inc1 = getelementptr i8, i8* %p_preinc, i64 2
  %a10 = load i8, i8* %p_postinc
  %a11 = load i8, i8* %p1_inc1
  %a12 = add i8 %a10, %a11
  store i8 %a12, i8* @dummy
  br i1 %pred, label %indirectbr, label %exit

op2:
; CHECK-LABEL: op2:
; CHECK-NEXT: %p2_inc = getelementptr i8, i8* %p_postinc, i64 1
  %p2_inc = getelementptr i8, i8* %p_preinc, i64 2
  %a2 = load i8, i8* %p_postinc
  store i8 %a2, i8* @dummy
  br i1 %pred, label %indirectbr, label %exit

op3:
  br i1 %pred, label %indirectbr, label %exit

indirectbr:
  %p_preinc = phi i8* [%p0, %entry], [%p1_inc2, %op1], [%p2_inc, %op2], [%p_postinc, %op3]
  %p_postinc = getelementptr i8, i8* %p_preinc, i64 1
  %next_op = load i8, i8* %p_preinc
  %p_zext = zext i8 %next_op to i64
  %slot = getelementptr [256 x i8*], [256 x i8*]* %table, i64 0, i64 %p_zext 
  %target = load i8*, i8** %slot
  indirectbr i8* %target, [label %exit, label %op1, label %op2]

exit:
  ret void
}
