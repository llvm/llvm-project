; REQUIRES: systemz-registered-target
; REQUIRES: target=s390x{{.*}}

; RUN: llc -mtriple=s390x-ibm-zos -filetype=obj %s -o %t.o
; RUN: llvm-nm --no-sort %t.o | FileCheck %s --check-prefix=TYPES
; RUN: llvm-nm --no-sort --print-size --defined-only %t.o | FileCheck %s --check-prefix=SIZES

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

; TYPES-DAG: {{^[0-9A-Fa-f]+}} D GlobalData
; TYPES-DAG: {{^[0-9A-Fa-f]+}} d LocalData
; TYPES-DAG: {{^[0-9A-Fa-f]+}} T GlobalFunc
; TYPES-DAG: {{^[0-9A-Fa-f]+}} t LocalFunc
; TYPES-DAG: {{^[0-9A-Fa-f]+}} T UseExternFunc
; TYPES-DAG: {{^ *}}U ExternFunc

; SIZES-DAG: {{^[0-9A-Fa-f]+}} {{0*4}} D GlobalData
; SIZES-DAG: {{^[0-9A-Fa-f]+}} {{0*4}} d LocalData
; SIZES-DAG: {{^[0-9A-Fa-f]+}} {{[0-9A-Fa-f]+}} T GlobalFunc
; SIZES-DAG: {{^[0-9A-Fa-f]+}} {{[0-9A-Fa-f]+}} t LocalFunc
; SIZES-DAG: {{^[0-9A-Fa-f]+}} {{[0-9A-Fa-f]+}} T UseExternFunc
