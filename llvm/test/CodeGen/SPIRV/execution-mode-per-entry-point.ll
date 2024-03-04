; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpMemoryModel
; CHECK-DAG: OpEntryPoint Kernel %[[#ENTRY1:]] "foo1"
; CHECK-DAG: OpEntryPoint Kernel %[[#ENTRY4:]] "foo4"
; CHECK-NOT: OpSource
; CHECK-DAG: OpExecutionMode %[[#ENTRY1]] {{[a-zA-Z]+}}
; CHECK-DAG: OpExecutionMode %[[#ENTRY4]] {{[a-zA-Z]+}}
; CHECK: OpSource

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECKN

; CHECKN: OpMemoryModel
; CHECKN-COUNT-2: OpEntryPoint Kernel
; CHECKN-NOT: OpEntryPoint Kernel
; CHECKN-COUNT-2: OpExecutionMode
; CHECKN-NOT: OpExecutionMode
; CHECKN: OpSource

define spir_kernel void @foo1() {
entry:
  ret void
}

define void @foo2() {
entry:
  ret void
}

define dso_local spir_func void @foo3() {
entry:
  ret void
}

define spir_kernel void @foo4() {
entry:
  ret void
}
