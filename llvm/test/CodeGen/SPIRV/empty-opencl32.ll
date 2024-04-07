; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; FIXME: ensure Magic Number, version number, generator's magic number, "bound" and "schema" are at least present

;; Ensure the required Capabilities are listed.
; CHECK-DAG: OpCapability Kernel
; CHECK-DAG: OpCapability Addresses

;; Ensure one, and only one, OpMemoryModel is defined.
; CHECK:     OpMemoryModel Physical32 OpenCL
; CHECK-NOT: OpMemoryModel
