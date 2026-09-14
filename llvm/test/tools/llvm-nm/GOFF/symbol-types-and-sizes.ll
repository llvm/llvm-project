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

; The functions within this test are label definitions (LD). The function entry
; point is typically created as a label within a container, and these label
; symbols do not have an explicit size set, hence why 0 is the expected size
; for the below functions.

; CHECK-DAG: {{^[0-9A-Fa-f]+}} 0000000000000004 D GlobalData
; CHECK-DAG: {{^[0-9A-Fa-f]+}} 0000000000000004 d LocalData
; CHECK-DAG: {{^[0-9A-Fa-f]+}} 0000000000000000 T GlobalFunc
; CHECK-DAG: {{^[0-9A-Fa-f]+}} 0000000000000000 t LocalFunc
; CHECK-DAG: {{^[0-9A-Fa-f]+}} 0000000000000000 T UseExternFunc
; CHECK-DAG:                                   U ExternFunc
