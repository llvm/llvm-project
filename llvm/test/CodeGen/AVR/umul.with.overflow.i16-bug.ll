; RUN: llc -O1 < %s -mtriple=avr | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9"

%Vs6UInt16 = type <{ i16 }>
%Sb = type <{ i1 }>

define hidden void @setServoAngle(i16) {
  ; CHECK-LABEL: entry
entry:
  %adjustedAngle = alloca %Vs6UInt16, align 2
  %1 = bitcast ptr %adjustedAngle to ptr
  %adjustedAngle._value = getelementptr inbounds %Vs6UInt16, ptr %adjustedAngle, i32 0, i32 0
  store i16 %0, ptr %adjustedAngle._value, align 2

;print(unsignedInt: adjustedAngle &* UInt16(11))
; breaks here
  %adjustedAngle._value2 = getelementptr inbounds %Vs6UInt16, ptr %adjustedAngle, i32 0, i32 0
  %2 = load i16, ptr %adjustedAngle._value2, align 2

; CHECK: mov r22, r24
; CHECK: mov r23, r25

; CHECK-DAG: ldi r20, 0
; CHECK-DAG: ldi r21, 0
; CHECK-DAG: ldi r18, 11
; CHECK-DAG: ldi r19, 0

; CHECK: mov r24, r20
; CHECK: mov r25, r21
; CHECK: call  __mulsi3
  %3 = call { i16, i1 } @llvm.umul.with.overflow.i16(i16 %2, i16 11)
  %4 = extractvalue { i16, i1 } %3, 0
  %5 = extractvalue { i16, i1 } %3, 1

  ; above code looks fine, how is it lowered?
  %6 = call i1 @printDefaultParam()
  call void @print(i16 %4, i1 %6)

; CHECK: ret
  ret void
}

declare void @print(i16, i1)
declare i1 @printDefaultParam()

; Function Attrs: nounwind readnone speculatable
declare { i16, i1 } @llvm.umul.with.overflow.i16(i16, i16)
