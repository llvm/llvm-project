; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

@TestA = alias void (), void ()* @TestC
@TestB = alias void (), void ()* @TestC
@TestC = alias void (), void ()* @TestD

define void @TestD() {
entry:
  ret void
}

; CHECK-LABEL: TestD:
; CHECK: TestC = TestD
; CHECK-DAG: TestB = TestC
; CHECK-DAG: TestA = TestC
