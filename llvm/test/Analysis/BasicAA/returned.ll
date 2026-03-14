; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct = type { i32, i32, i32 }

; CHECK-LABEL: test_simple

; CHECK-DAG: MustAlias: %struct* %st, %struct* %sta

; CHECK-DAG: MayAlias: %struct* %st, i32* %x
; CHECK-DAG: MayAlias: %struct* %st, i32* %y
; CHECK-DAG: MayAlias: %struct* %st, i32* %z

; CHECK-DAG: NoAlias: i32* %x, i32* %y
; CHECK-DAG: NoAlias: i32* %x, i32* %z
; CHECK-DAG: NoAlias: i32* %y, i32* %z

; CHECK-DAG: MayAlias: %struct* %st, %struct* %y
; CHECK-DAG: MayAlias: i32* %x, %struct* %y
; CHECK-DAG: MayAlias: i32* %x, i80* %y

; CHECK-DAG: MayAlias: %struct* %st, i64* %ya
; CHECK-DAG: MayAlias: i64* %ya, i32* %z
; CHECK-DAG: NoAlias: i32* %x, i64* %ya

; CHECK-DAG: MustAlias: %struct* %y, i32* %y
; CHECK-DAG: MustAlias: i32* %y, i64* %ya
; CHECK-DAG: MustAlias: i80* %y, i32* %y

define void @test_simple(ptr %st, i64 %i, i64 %j, i64 %k) {
  %x = getelementptr inbounds %struct, ptr %st, i64 %i, i32 0
  %y = getelementptr inbounds %struct, ptr %st, i64 %j, i32 1
  %sta = call ptr @func2(ptr %st)
  %z = getelementptr inbounds %struct, ptr %sta, i64 %k, i32 2
  %ya = call ptr @func1(ptr %y)
  load %struct, ptr %st
  load %struct, ptr %sta
  load i32, ptr %x
  load i32, ptr %y
  load i32, ptr %z
  load %struct, ptr %y
  load i80, ptr %y
  load i64, ptr %ya
  ret void
}

declare ptr @func1(ptr returned) nounwind
declare ptr @func2(ptr returned) nounwind

