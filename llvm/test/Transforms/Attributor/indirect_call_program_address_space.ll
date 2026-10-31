; RUN: opt -passes=attributor -S < %s | FileCheck %s

; Specializing an indirect call compares the called operand against each
; candidate callee. Functions live in the program address space, 9 here, so the
; comparison has to be made there: normalizing the operand to address space 0
; gave an icmp whose operands had different types.

target datalayout = "e-i64:64-G1-P9-A0"

@g = external addrspace(1) global i32

define internal void @cb1() {
; CHECK-LABEL: define internal void @cb1(
; CHECK-SAME: ) addrspace(9) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:    store i32 1, ptr addrspace(1) @g, align 4
; CHECK-NEXT:    ret void
;
  store i32 1, ptr addrspace(1) @g
  ret void
}

define internal void @cb2() {
; CHECK-LABEL: define internal void @cb2(
; CHECK-SAME: ) addrspace(9) #[[ATTR0]] {
; CHECK-NEXT:    store i32 2, ptr addrspace(1) @g, align 4
; CHECK-NEXT:    ret void
;
  store i32 2, ptr addrspace(1) @g
  ret void
}

define void @caller(i1 %c) {
; CHECK-LABEL: define void @caller(
; CHECK-SAME: i1 [[C:%.*]]) addrspace(9) #[[ATTR0]] {
; CHECK-NEXT:    [[FP:%.*]] = select i1 [[C]], ptr addrspace(9) @cb1, ptr addrspace(9) @cb2
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq ptr addrspace(9) [[FP]], @cb2
; CHECK-NEXT:    br i1 [[TMP1]], label %[[BB2:.*]], label %[[BB3:.*]]
; CHECK:       [[BB2]]:
; CHECK-NEXT:    call addrspace(9) void @cb2()
; CHECK-NEXT:    br label %[[BB6:.*]]
; CHECK:       [[BB3]]:
; CHECK-NEXT:    br i1 true, label %[[BB4:.*]], label %[[BB5:.*]]
; CHECK:       [[BB4]]:
; CHECK-NEXT:    call addrspace(9) void @cb1()
; CHECK-NEXT:    br label %[[BB6]]
; CHECK:       [[BB5]]:
; CHECK-NEXT:    unreachable
; CHECK:       [[BB6]]:
; CHECK-NEXT:    ret void
;
  %fp = select i1 %c, ptr addrspace(9) @cb1, ptr addrspace(9) @cb2
  call addrspace(9) void %fp()
  ret void
}
