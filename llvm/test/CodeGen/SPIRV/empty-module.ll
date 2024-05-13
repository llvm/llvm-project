; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-OCL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOOCL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:        OpCapability Linkage
; CHECK-NOOCL-DAG:  OpCapability Shader
; CHECK-OCL-DAG:      OpCapability Addresses
; CHECK-OCL-DAG:      OpCapability Kernel
; CHECK-OCL:          %1 = OpExtInstImport "OpenCL.std"
; CHECK-NOOCL:      OpMemoryModel Logical GLSL450
; CHECK-OCL:          OpMemoryModel Physical64 OpenCL
; CHECK-NOOCL:      OpSource Unknown 0
; CHECK-OCL:          OpSource OpenCL_CPP 100000
