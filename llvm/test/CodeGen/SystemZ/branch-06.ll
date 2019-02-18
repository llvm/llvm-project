; Test all condition-code masks that are relevant for CRJ.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @foo()
@g1 = global i16 0

define void @f1(i32 %target) {
; CHECK-LABEL: f1:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: crje %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp eq i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f2(i32 %target) {
; CHECK-LABEL: f2:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: crjlh %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp ne i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f3(i32 %target) {
; CHECK-LABEL: f3:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: crjle %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp sle i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f4(i32 %target) {
; CHECK-LABEL: f4:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: crjl %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp slt i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f5(i32 %target) {
; CHECK-LABEL: f5:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: crjh %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp sgt i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f6(i32 %target) {
; CHECK-LABEL: f6:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: crjhe %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp sge i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; Check that CRJ is used for checking equality with a zero-extending
; character load.
define void @f7(i8 *%targetptr) {
; CHECK-LABEL: f7:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: llc [[REG:%r[0-5]]],
; CHECK: crje %r2, [[REG]], .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %byte = load i8, i8 *%targetptr
  %target = zext i8 %byte to i32
  %cond = icmp eq i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; ...and zero-extending i16 loads.
define void @f8(i16 *%targetptr) {
; CHECK-LABEL: f8:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: llh [[REG:%r[0-5]]],
; CHECK: crje %r2, [[REG]], .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %half = load i16, i16 *%targetptr
  %target = zext i16 %half to i32
  %cond = icmp eq i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; ...unless the address is a global.
define void @f9(i16 *%targetptr) {
; CHECK-LABEL: f9:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: clhrl %r2, g1
; CHECK: je .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %half = load i16, i16 *@g1
  %target = zext i16 %half to i32
  %cond = icmp eq i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; Check that CRJ is used for checking order between two zero-extending
; byte loads, even if the original comparison was unsigned.
define void @f10(i8 *%targetptr1) {
; CHECK-LABEL: f10:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK-DAG: llc [[REG1:%r[0-5]]], 0(
; CHECK-DAG: llc [[REG2:%r[0-5]]], 1(
; CHECK: crjl [[REG1]], [[REG2]], .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %targetptr2 = getelementptr i8, i8 *%targetptr1, i64 1
  %byte1 = load i8, i8 *%targetptr1
  %byte2 = load i8, i8 *%targetptr2
  %ext1 = zext i8 %byte1 to i32
  %ext2 = zext i8 %byte2 to i32
  %cond = icmp ult i32 %ext1, %ext2
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; ...likewise halfword loads.
define void @f11(i16 *%targetptr1) {
; CHECK-LABEL: f11:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK-DAG: llh [[REG1:%r[0-5]]], 0(
; CHECK-DAG: llh [[REG2:%r[0-5]]], 2(
; CHECK: crjl [[REG1]], [[REG2]], .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %targetptr2 = getelementptr i16, i16 *%targetptr1, i64 1
  %half1 = load i16, i16 *%targetptr1
  %half2 = load i16, i16 *%targetptr2
  %ext1 = zext i16 %half1 to i32
  %ext2 = zext i16 %half2 to i32
  %cond = icmp ult i32 %ext1, %ext2
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
