; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

@TestA = alias void (), ptr @TestC
@TestB = alias void (), ptr @TestC
@TestC = alias void (), ptr @TestD

define void @TestD() {
entry:
  ret void
}

; CHECK-LABEL: TestD:
; CHECK: .set TestC, TestD
; CHECK-DAG: .set TestB, TestC
; CHECK-DAG: .set TestA, TestC
