; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; Ensure the required Capabilities are listed.
; CHECK-DAG: OpCapability Kernel
; CHECK-DAG: OpCapability Addresses

;; Ensure one, and only one, OpMemoryModel is defined.
; CHECK:     OpMemoryModel Physical32 OpenCL
; CHECK-NOT: OpMemoryModel
