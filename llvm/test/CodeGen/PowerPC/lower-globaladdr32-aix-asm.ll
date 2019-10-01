; RUN: llc -mtriple powerpc-ibm-aix-xcoff \
; RUN: -code-model=small < %s | FileCheck %s

@b = common global i32 0
@a = common global i32 0

define void @test() {
  %1 = load i32, i32* @b
  store i32 %1, i32* @a
  ret void
}

; CHECK-LABEL:   test
; CHECK-DAG:       lwz [[REG1:[0-9]+]], LC0(2)
; CHECK-DAG:       lwz [[REG2:[0-9]+]], LC1(2)
; CHECK-DAG:       lwz [[REG3:[0-9]+]], 0([[REG1]])
; CHECK:           stw [[REG3]], 0([[REG2]])
; CHECK:           blr

; TODO Update test when TOC-entry emission lands.
; CHECK-NOT: .tc
