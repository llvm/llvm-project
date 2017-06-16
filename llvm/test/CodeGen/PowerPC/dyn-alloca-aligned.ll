; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.s = type { i32, i32 }

declare void @bar(i32*, i32*) #0

define void @goo(%struct.s* byval nocapture readonly %a, i32 signext %n) #0 {
entry:
  %0 = zext i32 %n to i64
  %vla = alloca i32, i64 %0, align 128
  %vla1 = alloca i32, i64 %0, align 128
  %a2 = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 0
  %1 = load i32, i32* %a2, align 4
  store i32 %1, i32* %vla1, align 128
  %b = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 1
  %2 = load i32, i32* %b, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %vla1, i64 1
  store i32 %2, i32* %arrayidx3, align 4
  call void @bar(i32* %vla1, i32* %vla) #0
  ret void

; CHECK-LABEL: @goo

; CHECK-DAG: li [[REG1:[0-9]+]], -128
; CHECK-DAG: neg [[REG2:[0-9]+]],
; CHECK: and [[REG3:[0-9]+]], [[REG2]], [[REG1]]
; CHECK: stdux {{[0-9]+}}, 1, [[REG3]]

; CHECK: blr

}

attributes #0 = { nounwind }
