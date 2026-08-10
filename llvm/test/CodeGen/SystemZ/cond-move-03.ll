; Test LOCFHR and LOCHHI.
; See comments in asm-18.ll about testing high-word operations.
;
; RUN: llc < %s -verify-machineinstrs -mtriple=s390x-linux-gnu -mcpu=z13 \
; RUN:   -no-integrated-as | FileCheck %s
;
; Run the test again to make sure it still works the same even
; in the presence of the select instructions.
; RUN: llc < %s -verify-machineinstrs -mtriple=s390x-linux-gnu -mcpu=z15 \
; RUN:   -no-integrated-as | FileCheck %s

define void @f1(i32 %limit) {
; CHECK-LABEL: f1:
; CHECK-DAG: stepa [[REG1:%r[0-5]]]
; CHECK-DAG: stepb [[REG2:%r[0-5]]]
; CHECK-DAG: clfi %r2, 42
; CHECK: locfhrhe [[REG1]], [[REG2]]
; CHECK: stepc [[REG1]]
; CHECK: br %r14
  %a = call i32 asm sideeffect "stepa $0", "=h"()
  %b = call i32 asm sideeffect "stepb $0", "=h"()
  %cond = icmp ult i32 %limit, 42
  %res = select i1 %cond, i32 %a, i32 %b
  call void asm sideeffect "stepc $0", "h"(i32 %res)
  call void asm sideeffect "use $0", "h"(i32 %b)
  ret void
}

define void @f2(i32 %limit) {
; CHECK-LABEL: f2:
; CHECK-DAG: stepa [[REG1:%r[0-5]]]
; CHECK-DAG: stepb [[REG2:%r[0-5]]]
; CHECK-DAG: clijl %r2, 42, [[LABEL:.LBB[0-9_]+]]
; CHECK: risbhg [[REG1]], [[REG2]], 0, 159, 32
; CHECK: [[LABEL]]
; CHECK: stepc [[REG1]]
; CHECK: br %r14
  %dummy = call i32 asm sideeffect "dummy $0", "=h"()
  %a = call i32 asm sideeffect "stepa $0", "=h"()
  %b = call i32 asm sideeffect "stepb $0", "=r"()
  %cond = icmp ult i32 %limit, 42
  %res = select i1 %cond, i32 %a, i32 %b
  call void asm sideeffect "stepc $0", "h"(i32 %res)
  call void asm sideeffect "dummy $0", "h"(i32 %dummy)
  call void asm sideeffect "use $0", "r"(i32 %b)
  ret void
}

define void @f3(i32 %limit) {
; CHECK-LABEL: f3:
; CHECK-DAG: stepa [[REG1:%r[0-5]]]
; CHECK-DAG: stepb [[REG2:%r[0-5]]]
; CHECK-DAG: clijhe %r2, 42, [[LABEL:.LBB[0-9_]+]]
; CHECK: risbhg [[REG2]], [[REG1]], 0, 159, 32
; CHECK: [[LABEL]]
; CHECK: stepc [[REG2]]
; CHECK: br %r14
  %dummy = call i32 asm sideeffect "dummy $0", "=h"()
  %a = call i32 asm sideeffect "stepa $0", "=r"()
  %b = call i32 asm sideeffect "stepb $0", "=h"()
  %cond = icmp ult i32 %limit, 42
  %res = select i1 %cond, i32 %a, i32 %b
  call void asm sideeffect "stepc $0", "h"(i32 %res)
  call void asm sideeffect "dummy $0", "h"(i32 %dummy)
  call void asm sideeffect "use $0", "r"(i32 %a)
  ret void
}

define void @f4(i32 %limit) {
; CHECK-LABEL: f4:
; CHECK-DAG: stepa [[REG1:%r[0-5]]]
; CHECK-DAG: stepb [[REG2:%r[0-5]]]
; CHECK-DAG: clijl %r2, 42, [[LABEL:.LBB[0-9_]+]]
; CHECK: risblg [[REG1]], [[REG2]], 0, 159, 32
; CHECK: [[LABEL]]
; CHECK: stepc [[REG1]]
; CHECK: br %r14
  %dummy = call i32 asm sideeffect "dummy $0", "=h"()
  %a = call i32 asm sideeffect "stepa $0", "=r"()
  %b = call i32 asm sideeffect "stepb $0", "=h"()
  %cond = icmp ult i32 %limit, 42
  %res = select i1 %cond, i32 %a, i32 %b
  call void asm sideeffect "stepc $0", "r"(i32 %res)
  call void asm sideeffect "dummy $0", "h"(i32 %dummy)
  call void asm sideeffect "use $0", "h"(i32 %b)
  ret void
}

define void @f5(i32 %limit) {
; CHECK-LABEL: f5:
; CHECK-DAG: stepa [[REG2:%r[0-5]]]
; CHECK-DAG: stepb [[REG1:%r[0-5]]]
; CHECK-DAG: clijhe %r2, 42, [[LABEL:.LBB[0-9_]+]]
; CHECK: risblg [[REG1]], [[REG2]], 0, 159, 32
; CHECK: [[LABEL]]
; CHECK: stepc [[REG1]]
; CHECK: br %r14
  %dummy = call i32 asm sideeffect "dummy $0", "=h"()
  %a = call i32 asm sideeffect "stepa $0", "=h"()
  %b = call i32 asm sideeffect "stepb $0", "=r"()
  %cond = icmp ult i32 %limit, 42
  %res = select i1 %cond, i32 %a, i32 %b
  call void asm sideeffect "stepc $0", "r"(i32 %res)
  call void asm sideeffect "dummy $0", "h"(i32 %dummy)
  ret void
}

; Check that we also get LOCFHR as a result of early if-conversion.
define void @f6(i32 %limit) {
; CHECK-LABEL: f6:
; CHECK-DAG: stepa [[REG1:%r[0-5]]]
; CHECK-DAG: stepb [[REG2:%r[0-5]]]
; CHECK-DAG: clfi %r2, 41
; CHECK: locfhrh [[REG1]], [[REG2]]
; CHECK: stepc [[REG1]]
; CHECK: br %r14
entry:
  %a = call i32 asm sideeffect "stepa $0", "=h"()
  %b = call i32 asm sideeffect "stepb $0", "=h"()
  %cond = icmp ult i32 %limit, 42
  br i1 %cond, label %if.then, label %return

if.then:
  br label %return

return:
  %res = phi i32 [ %a, %if.then ], [ %b, %entry ]
  call void asm sideeffect "stepc $0", "h"(i32 %res)
  call void asm sideeffect "use $0", "h"(i32 %b)
  ret void
}

; Check that inverting the condition works as well.
define void @f7(i32 %limit) {
; CHECK-LABEL: f7:
; CHECK-DAG: stepa [[REG1:%r[0-5]]]
; CHECK-DAG: stepb [[REG2:%r[0-5]]]
; CHECK-DAG: clfi %r2, 41
; CHECK: locfhrle [[REG1]], [[REG2]]
; CHECK: stepc [[REG1]]
; CHECK: br %r14
entry:
  %a = call i32 asm sideeffect "stepa $0", "=h"()
  %b = call i32 asm sideeffect "stepb $0", "=h"()
  %cond = icmp ult i32 %limit, 42
  br i1 %cond, label %if.then, label %return

if.then:
  br label %return

return:
  %res = phi i32 [ %b, %if.then ], [ %a, %entry ]
  call void asm sideeffect "stepc $0", "h"(i32 %res)
  call void asm sideeffect "use $0", "h"(i32 %b)
  ret void
}

define void @f8(i32 %limit) {
; CHECK-LABEL: f8:
; CHECK: clfi %r2, 42
; CHECK: lochhil [[REG:%r[0-5]]], 32767
; CHECK: stepa [[REG]]
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %res = select i1 %cond, i32 32767, i32 0
  call void asm sideeffect "stepa $0", "h"(i32 %res)
  ret void
}

define void @f9(i32 %limit) {
; CHECK-LABEL: f9:
; CHECK: clfi %r2, 42
; CHECK: lochhil [[REG:%r[0-5]]], -32768
; CHECK: stepa [[REG]]
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %res = select i1 %cond, i32 -32768, i32 0
  call void asm sideeffect "stepa $0", "h"(i32 %res)
  ret void
}

; Check that we also get LOCHHI as a result of early if-conversion.
define void @f10(i32 %limit) {
; CHECK-LABEL: f10:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: clfi %r2, 41
; CHECK: lochhile [[REG]], 123
; CHECK: stepb [[REG]]
; CHECK: br %r14
entry:
  %a = call i32 asm sideeffect "stepa $0", "=h"()
  %cond = icmp ult i32 %limit, 42
  br i1 %cond, label %if.then, label %return

if.then:
  br label %return

return:
  %res = phi i32 [ 123, %if.then ], [ %a, %entry ]
  call void asm sideeffect "stepb $0", "h"(i32 %res)
  ret void
}

; Check that inverting the condition works as well.
define void @f11(i32 %limit) {
; CHECK-LABEL: f11:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: clfi %r2, 41
; CHECK: lochhih [[REG]], 123
; CHECK: stepb [[REG]]
; CHECK: br %r14
entry:
  %a = call i32 asm sideeffect "stepa $0", "=h"()
  %cond = icmp ult i32 %limit, 42
  br i1 %cond, label %if.then, label %return

if.then:
  br label %return

return:
  %res = phi i32 [ %a, %if.then ], [ 123, %entry ]
  call void asm sideeffect "stepb $0", "h"(i32 %res)
  ret void
}
