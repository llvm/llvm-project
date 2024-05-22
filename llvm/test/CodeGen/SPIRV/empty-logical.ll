; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s

;; Ensure the required Capabilities are listed.
; CHECK-DAG: OpCapability Shader
; CHECK-DAG: OpCapability Linkage

;; Ensure one, and only one, OpMemoryModel is defined.
; CHECK:     OpMemoryModel Logical GLSL450
; CHECK-NOT: OpMemoryModel
