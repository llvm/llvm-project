; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -mattr=+micromips -verify-machineinstrs < %s | FileCheck %s

define i32 @f1() {
entry:
; CHECK-LABEL: f1:
; CHECK: addiusp
; CHECK: addiur1sp
; CHECK: addiusp
  %a = alloca [10 x i32], align 4
  call void @init(ptr %a)
  %0 = load i32, ptr %a, align 4
  ret i32 %0
}

declare void @init(ptr)

