; REQUIRES: systemz-registered-target

; RUN: llc -mtriple=s390x-ibm-zos -filetype=obj %s -o %t.o
; RUN: llvm-nm --no-sort --print-size %t.o | FileCheck %s

target triple = "s390x-ibm-zos"

@GlobalData = global i32 42, align 4
@LocalData = internal global i32 7, align 4

declare void @ExternFunc()

define void @GlobalFunc() {
entry:
  ret void
}

define internal void @LocalFunc() {
entry:
  ret void
}

define void @UseExternFunc() {
entry:
  call void @ExternFunc()
  ret void
}

; CHECK-DAG: {{^[0-9A-Fa-f]+}} {{0*4}} D GlobalData
; CHECK-DAG: {{^[0-9A-Fa-f]+}} {{0*4}} d LocalData
; CHECK-DAG: {{^[0-9A-Fa-f]+}} {{[0-9A-Fa-f]+}} T GlobalFunc
; CHECK-DAG: {{^[0-9A-Fa-f]+}} {{[0-9A-Fa-f]+}} t LocalFunc
; CHECK-DAG: {{^[0-9A-Fa-f]+}} {{[0-9A-Fa-f]+}} T UseExternFunc
; CHECK-DAG: {{^ *}}U ExternFunc
