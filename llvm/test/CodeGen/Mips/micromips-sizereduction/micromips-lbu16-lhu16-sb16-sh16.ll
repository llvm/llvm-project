; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=+micromips -verify-machineinstrs < %s | FileCheck %s

define void @f1(ptr %p) {
entry:
; CHECK-LABEL: f1:
; CHECK: lbu16
; CHECK: sb16
  %0 = load i8, ptr %p, align 4
  %a = zext i8 %0 to i32
  %and = and i32 %a, 1
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i8 0, ptr %p, align 1
  br label %if.end

if.end:
  ret void
}

define void @f2(ptr %p) {
entry:
; CHECK-LABEL: f2:
; CHECK: lhu16
; CHECK: sh16
  %0 = load i16, ptr %p, align 2
  %a = zext i16 %0 to i32
  %and = and i32 %a, 2
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i16 0, ptr %p, align 2
  br label %if.end

if.end:
  ret void
}

