; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=+micromips -verify-machineinstrs < %s | FileCheck %s

define i32 @f1() {
entry:
; CHECK-LABEL: f1:
; CHECK: addiusp
; CHECK: addiur1sp
; CHECK: addiusp
  %a = alloca [10 x i32], align 4
  %index = getelementptr inbounds [10 x i32], [10 x i32]* %a, i32 0, i32 0
  call void @init(i32* %index)
  %0 = load i32, i32* %index, align 4
  ret i32 %0
}

declare void @init(i32*)

