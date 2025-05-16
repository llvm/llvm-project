; Check names and decoration of global variables.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s -check-prefixes=CHECK,OCL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s -check-prefixes=CHECK,OCL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s -check-prefixes=CHECK,VK
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#id18:]] "G1"
; CHECK-DAG: OpName %[[#id22:]] "g1"
; CHECK-DAG: OpName %[[#id23:]] "g2"
; CHECK-DAG: OpName %[[#id27:]] "g4"
; CHECK-DAG: OpName %[[#id30:]] "c1"
; CHECK-DAG: OpName %[[#id31:]] "n_t"
; CHECK-DAG: OpName %[[#id32:]] "w"
; CHECK-DAG: OpName %[[#id34:]] "a.b"
; CHECK-DAG: OpName %[[#id35:]] "e"
; CHECK-DAG: OpName %[[#id36:]] "y.z"
; CHECK-DAG: OpName %[[#id38:]] "x"

; CHECK-NOT: OpDecorate %[[#id18]] LinkageAttributes
; OCL-DAG: OpDecorate %[[#id18]] Constant
; OCL-DAG: OpDecorate %[[#id22]] Alignment 4
; VK-NOT: OpDecorate {{.*}} Constant
; VK-NOT: OpDecorate {{.*}} Alignment
; CHECK-DAG: OpDecorate %[[#id22]] LinkageAttributes "g1" Export
; OCL-DAG: OpDecorate %[[#id23]] Alignment 4
; OCL-DAG: OpDecorate %[[#id27]] Alignment 4
; VK-NOT: OpDecorate {{.*}} Constant
; VK-NOT: OpDecorate {{.*}} Alignment
; CHECK-DAG: OpDecorate %[[#id27]] LinkageAttributes "g4" Export
; OCL-DAG: OpDecorate %[[#id30]] Constant
; OCL-DAG: OpDecorate %[[#id30]] Alignment 4
; VK-NOT: OpDecorate {{.*}} Constant
; VK-NOT: OpDecorate {{.*}} Alignment
; CHECK-DAG: OpDecorate %[[#id30]] LinkageAttributes "c1" Export
; OCL-DAG: OpDecorate %[[#id31]] Constant
; VK-NOT: OpDecorate {{.*}} Constant
; VK-NOT: OpDecorate {{.*}} Alignment
; CHECK-DAG: OpDecorate %[[#id31]] LinkageAttributes "n_t" Import
; OCL-DAG: OpDecorate %[[#id32]] Constant
; OCL-DAG: OpDecorate %[[#id32]] Alignment 4
; VK-NOT: OpDecorate {{.*}} Constant
; VK-NOT: OpDecorate {{.*}} Alignment
; CHECK-DAG: OpDecorate %[[#id32]] LinkageAttributes "w" Export
; OCL-DAG: OpDecorate %[[#id34]] Constant
; OCL-DAG: OpDecorate %[[#id34]] Alignment 4
; VK-NOT: OpDecorate {{.*}} Constant
; VK-NOT: OpDecorate {{.*}} Alignment
; CHECK-DAG: OpDecorate %[[#id35]] LinkageAttributes "e" Import
; OCL-DAG: OpDecorate %[[#id36]] Alignment 4
; OCL-DAG: OpDecorate %[[#id38]] Constant
; OCL-DAG: OpDecorate %[[#id38]] Alignment 4
; VK-NOT: OpDecorate {{.*}} Constant
; VK-NOT: OpDecorate {{.*}} Alignment

%"class.sycl::_V1::nd_item" = type { i8 }

@G1 = private unnamed_addr addrspace(1) constant %"class.sycl::_V1::nd_item" poison, align 1
@g1 = addrspace(1) global i32 1, align 4
@g2 = internal addrspace(1) global i32 2, align 4
@g4 = common addrspace(1) global i32 0, align 4
@c1 = addrspace(2) constant [2 x i32] [i32 0, i32 1], align 4
@n_t = external addrspace(2) constant [256 x i32]
@w = addrspace(1) constant i32 0, align 4
@a.b = internal addrspace(2) constant [2 x i32] [i32 2, i32 3], align 4
@e = external addrspace(1) global i32
@y.z = internal addrspace(1) global i32 0, align 4
@x = internal addrspace(2) constant float 1.000000e+00, align 4

define internal spir_func void @foo() {
  ret void
}
