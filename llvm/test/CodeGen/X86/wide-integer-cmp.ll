; RUN: llc -mtriple=i686-linux-gnu %s -o - | FileCheck %s


define i32 @branch_eq(i64 %a, i64 %b) {
entry:
  %cmp = icmp eq i64 %a, %b
	br i1 %cmp, label %bb1, label %bb2
bb1:
  ret i32 1
bb2:
  ret i32 2

; CHECK-LABEL: branch_eq:
; CHECK: movl 4(%esp), [[LHSLo:%[a-z]+]]
; CHECK: movl 8(%esp), [[LHSHi:%[a-z]+]]
; CHECK: xorl 16(%esp), [[LHSHi]]
; CHECK: xorl 12(%esp), [[LHSLo]]
; CHECK: orl [[LHSHi]], [[LHSLo]]
; CHECK: jne [[FALSE:.LBB[0-9_]+]]
; CHECK: movl $1, %eax
; CHECK: retl
; CHECK: [[FALSE]]:
; CHECK: movl $2, %eax
; CHECK: retl
}

define i32 @branch_slt(i64 %a, i64 %b) {
entry:
  %cmp = icmp slt i64 %a, %b
	br i1 %cmp, label %bb1, label %bb2
bb1:
  ret i32 1
bb2:
  ret i32 2

; CHECK-LABEL: branch_slt:
; CHECK: movl 4(%esp), [[LHSLo:%[a-z]+]]
; CHECK: movl 8(%esp), [[LHSHi:%[a-z]+]]
; CHECK: cmpl 12(%esp), [[LHSLo]]
; CHECK: sbbl 16(%esp), [[LHSHi]]
; CHECK: jge [[FALSE:.LBB[0-9_]+]]
; CHECK: movl $1, %eax
; CHECK: retl
; CHECK: [[FALSE]]:
; CHECK: movl $2, %eax
; CHECK: retl
}

define i32 @branch_ule(i64 %a, i64 %b) {
entry:
  %cmp = icmp ule i64 %a, %b
	br i1 %cmp, label %bb1, label %bb2
bb1:
  ret i32 1
bb2:
  ret i32 2

; CHECK-LABEL: branch_ule:
; CHECK: movl 12(%esp), [[RHSLo:%[a-z]+]]
; CHECK: movl 16(%esp), [[RHSHi:%[a-z]+]]
; CHECK: cmpl 4(%esp), [[RHSLo]]
; CHECK: sbbl 8(%esp), [[RHSHi]]
; CHECK: jb [[FALSE:.LBB[0-9_]+]]
; CHECK: movl $1, %eax
; CHECK: retl
; CHECK: [[FALSE]]:
; CHECK: movl $2, %eax
; CHECK: retl
}

define i32 @set_gt(i64 %a, i64 %b) {
entry:
  %cmp = icmp sgt i64 %a, %b
  %res = select i1 %cmp, i32 1, i32 0
  ret i32 %res

; CHECK-LABEL: set_gt:
; CHECK: movl 12(%esp), [[RHSLo:%[a-z]+]]
; CHECK: movl 16(%esp), [[RHSHi:%[a-z]+]]
; CHECK: cmpl 4(%esp), [[RHSLo]]
; CHECK: sbbl 8(%esp), [[RHSHi]]
; CHECK: setl %al
; CHECK: retl
}

define i32 @test_wide(i128 %a, i128 %b) {
entry:
  %cmp = icmp slt i128 %a, %b
	br i1 %cmp, label %bb1, label %bb2
bb1:
  ret i32 1
bb2:
  ret i32 2

; CHECK-LABEL: test_wide:
; CHECK: cmpl 24(%esp)
; CHECK: sbbl 28(%esp)
; CHECK: sbbl 32(%esp)
; CHECK: sbbl 36(%esp)
; CHECK: jge [[FALSE:.LBB[0-9_]+]]
; CHECK: movl $1, %eax
; CHECK: retl
; CHECK: [[FALSE]]:
; CHECK: movl $2, %eax
; CHECK: retl
}

define i32 @test_carry_false(i64 %a, i64 %b) {
entry:
  %x = and i64 %a, -4294967296 ;0xffffffff00000000
  %y = and i64 %b, -4294967296
  %cmp = icmp slt i64 %x, %y
	br i1 %cmp, label %bb1, label %bb2
bb1:
  ret i32 1
bb2:
  ret i32 2

; The comparison of the low bits will be folded to a CARRY_FALSE node. Make
; sure the code can handle that.
; CHECK-LABEL: carry_false:
; CHECK: movl 8(%esp), [[LHSHi:%[a-z]+]]
; CHECK: cmpl 16(%esp), [[LHSHi]]
; CHECK: jge [[FALSE:.LBB[0-9_]+]]
; CHECK: movl $1, %eax
; CHECK: retl
; CHECK: [[FALSE]]:
; CHECK: movl $2, %eax
; CHECK: retl
}
