; RUN: opt -debugify-each -O3 -S -o - < %s | FileCheck %s
; RUN: opt -debugify-each -instrprof -sroa -sccp -S -o - < %s | FileCheck %s

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

; Verify that the module & function (check-)debugify passes run at least twice.

; CHECK-DAG: CheckModuleDebugify: PASS
; CHECK-DAG: CheckFunctionDebugify: PASS
; CHECK-DAG: CheckFunctionDebugify: PASS
; CHECK-DAG: CheckFunctionDebugify: PASS
; CHECK-DAG: CheckFunctionDebugify: PASS

; CHECK-DAG: CheckModuleDebugify: PASS
; CHECK-DAG: CheckFunctionDebugify: PASS
; CHECK-DAG: CheckFunctionDebugify: PASS
; CHECK-DAG: CheckFunctionDebugify: PASS
; CHECK-DAG: CheckFunctionDebugify: PASS
