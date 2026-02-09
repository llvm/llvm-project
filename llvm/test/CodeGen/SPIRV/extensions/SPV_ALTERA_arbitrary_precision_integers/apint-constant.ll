; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - | FileCheck %s

; Verify that wide integer constants (>64 bits) are correctly encoded as
; OpConstant with multi-word literals.

; CHECK-DAG: %[[#INT128:]] = OpTypeInt 128 0
; CHECK-DAG: %[[#INT96:]] = OpTypeInt 96 0
; CHECK-DAG: %[[#INT97:]] = OpTypeInt 97 0
; CHECK-DAG: %[[#NEG128:]] = OpConstant %[[#INT128]] 4294965247 4294967295 4294967295 4294967295
; CHECK-DAG: %[[#ONE128:]] = OpConstant %[[#INT128]] 1 0 0 0
; CHECK-DAG: %[[#BOUNDARY:]] = OpConstant %[[#INT128]] 4294967295 4294967295 0 0
; CHECK-DAG: %[[#ZERO128:]] = OpConstantNull %[[#INT128]]
; CHECK-DAG: %[[#NEG96:]] = OpConstant %[[#INT96]] 4294967295 4294967295 4294967295
; CHECK-DAG: %[[#OVER64:]] = OpConstant %[[#INT96]] 1 0 1
; CHECK-DAG: %[[#NEG97:]] = OpConstant %[[#INT97]] 4294967295 4294967295 4294967295 1
; CHECK-DAG: %[[#OVER64_I97:]] = OpConstant %[[#INT97]] 1 0 1 0
; CHECK-DAG: %[[#I97_MAX:]] = OpConstant %[[#INT97]] 0 0 0 1

; CHECK: OpStore %[[#]] %[[#NEG128]] Aligned 16
; CHECK: OpStore %[[#]] %[[#ONE128]] Aligned 16
; CHECK: OpStore %[[#]] %[[#BOUNDARY]] Aligned 16
; CHECK: OpStore %[[#]] %[[#ZERO128]] Aligned 16

; CHECK: OpStore %[[#]] %[[#NEG96]] Aligned 16
; CHECK: OpStore %[[#]] %[[#OVER64]] Aligned 16

; CHECK: OpStore %[[#]] %[[#NEG97]] Aligned 16
; CHECK: OpStore %[[#]] %[[#OVER64_I97]] Aligned 16
; CHECK: OpStore %[[#]] %[[#I97_MAX]] Aligned 16

define spir_func void @test_i128_const(ptr addrspace(4) %p) addrspace(4) {
entry:
  store i128 -2049, ptr addrspace(4) %p, align 16
  store i128 1, ptr addrspace(4) %p, align 16
  store i128 18446744073709551615, ptr addrspace(4) %p, align 16
  store i128 0, ptr addrspace(4) %p, align 16
  ret void
}

define spir_func void @test_i96_const(ptr addrspace(4) %p) addrspace(4) {
entry:
  store i96 -1, ptr addrspace(4) %p, align 16
  store i96 18446744073709551617, ptr addrspace(4) %p, align 16
  ret void
}

define spir_func void @test_i97_const(ptr addrspace(4) %p) addrspace(4) {
entry:
  store i97 -1, ptr addrspace(4) %p, align 16
  store i97 18446744073709551617, ptr addrspace(4) %p, align 16
  store i97 79228162514264337593543950336, ptr addrspace(4) %p, align 16
  ret void
}
