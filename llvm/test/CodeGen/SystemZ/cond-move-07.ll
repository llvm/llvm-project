; Test SELFHR.
; See comments in asm-18.ll about testing high-word operations.
;
; RUN: llc < %s -verify-machineinstrs -mtriple=s390x-linux-gnu -mcpu=z15 \
; RUN:   -no-integrated-as | FileCheck %s

define void @f1(i32 %limit) {
; CHECK-LABEL: f1:
; CHECK-DAG: stepa [[REG1:%r[0-5]]]
; CHECK-DAG: stepb [[REG2:%r[0-5]]]
; CHECK-DAG: clfi %r2, 42
; CHECK: selfhrl [[REG3:%r[0-5]]], [[REG1]], [[REG2]]
; CHECK: stepc [[REG3]]
; CHECK: br %r14
  %a = call i32 asm sideeffect "stepa $0", "=h"()
  %b = call i32 asm sideeffect "stepb $0", "=h"()
  %cond = icmp ult i32 %limit, 42
  %res = select i1 %cond, i32 %a, i32 %b
  call void asm sideeffect "stepc $0", "h"(i32 %res)
  call void asm sideeffect "use $0", "h"(i32 %a)
  call void asm sideeffect "use $0", "h"(i32 %b)
  ret void
}

; Check that we also get SELFHR as a result of early if-conversion.
define void @f2(i32 %limit) {
; CHECK-LABEL: f2:
; CHECK-DAG: stepa [[REG1:%r[0-5]]]
; CHECK-DAG: stepb [[REG2:%r[0-5]]]
; CHECK-DAG: clfi %r2, 41
; CHECK: selfhrh [[REG3:%r[0-5]]], [[REG2]], [[REG1]]
; CHECK: stepc [[REG3]]
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
  call void asm sideeffect "use $0", "h"(i32 %a)
  call void asm sideeffect "use $0", "h"(i32 %b)
  ret void
}

; Check that inverting the condition works as well.
define void @f3(i32 %limit) {
; CHECK-LABEL: f3:
; CHECK-DAG: stepa [[REG1:%r[0-5]]]
; CHECK-DAG: stepb [[REG2:%r[0-5]]]
; CHECK-DAG: clfi %r2, 41
; CHECK: selfhrh [[REG3:%r[0-5]]], [[REG1]], [[REG2]]
; CHECK: stepc [[REG3]]
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
  call void asm sideeffect "use $0", "h"(i32 %a)
  call void asm sideeffect "use $0", "h"(i32 %b)
  ret void
}

