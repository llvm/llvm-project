; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-linux %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-linux %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#i32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#mask24:]] = OpConstant %[[#i32]] 16777215
; CHECK-DAG: %[[#i64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#mask40:]] = OpConstant %[[#i64]] 1099511627775

; CHECK: %[[#]] = OpBitCount %[[#]] %[[#]]
; CHECK: %[[#]] = OpBitCount %[[#]] %[[#]]
; CHECK: %[[#]] = OpBitCount %[[#]] %[[#]]
; CHECK: %[[#]] = OpBitCount %[[#]] %[[#]]
; CHECK: %[[#]] = OpBitCount %[[#]] %[[#]]

@g1 = addrspace(1) global i8 undef, align 4
@g2 = addrspace(1) global i16 undef, align 4
@g3 = addrspace(1) global i32 undef, align 4
@g4 = addrspace(1) global i64 undef, align 8
@g5 = addrspace(1) global <2 x i32> undef, align 4


define dso_local spir_kernel void @test(i8 %x8, i16 %x16, i32 %x32, i64 %x64, <2 x i32> %x2i32) local_unnamed_addr {
entry:
  %0 = tail call i8 @llvm.ctpop.i8(i8 %x8)
  store i8 %0, ptr addrspace(1) @g1, align 4
  %1 = tail call i16 @llvm.ctpop.i16(i16 %x16)
  store i16 %1, ptr addrspace(1) @g2, align 4
  %2 = tail call i32 @llvm.ctpop.i32(i32 %x32)
  store i32 %2, ptr addrspace(1) @g3, align 4
  %3 = tail call i64 @llvm.ctpop.i64(i64 %x64)
  store i64 %3, ptr addrspace(1) @g4, align 8
  %4 = tail call <2 x i32> @llvm.ctpop.v2i32(<2 x i32> %x2i32)
  store <2 x i32> %4, ptr addrspace(1) @g5, align 4

  ret void
}

; CHECK: %[[#]] = OpFunction
; CHECK: %[[#sum:]] = OpIAdd %[[#i32]]
; CHECK: %[[#masked24:]] = OpBitwiseAnd %[[#i32]] %[[#sum]] %[[#mask24]]
; CHECK: %[[#count24:]] = OpBitCount %[[#i32]] %[[#masked24]]
; CHECK: OpReturnValue %[[#count24]]

define spir_func i24 @test_i24(i32 %x) {
  %narrow = trunc i32 %x to i24
  %sum = add i24 %narrow, 1
  %count = tail call i24 @llvm.ctpop.i24(i24 %sum)
  ret i24 %count
}

; The high bits of an i24 parameter are genuinely undefined, so the mask is
; what makes the count correct.
; CHECK: %[[#]] = OpFunction
; CHECK: %[[#maskedarg:]] = OpBitwiseAnd %[[#i32]] %[[#]] %[[#mask24]]
; CHECK: %[[#countarg:]] = OpBitCount %[[#i32]] %[[#maskedarg]]
; CHECK: OpReturnValue %[[#countarg]]

define spir_func i24 @test_i24_arg(i24 %x) {
  %count = tail call i24 @llvm.ctpop.i24(i24 %x)
  ret i24 %count
}

; CHECK: %[[#]] = OpFunction
; CHECK: %[[#masked40:]] = OpBitwiseAnd %[[#i64]] %[[#]] %[[#mask40]]
; CHECK: %[[#count40:]] = OpBitCount %[[#i64]] %[[#masked40]]
; CHECK: OpReturnValue %[[#count40]]

define spir_func i40 @test_i40_arg(i40 %x) {
  %count = tail call i40 @llvm.ctpop.i40(i40 %x)
  ret i40 %count
}

declare i8 @llvm.ctpop.i8(i8)

declare i16 @llvm.ctpop.i16(i16)

declare i24 @llvm.ctpop.i24(i24)

declare i40 @llvm.ctpop.i40(i40)

declare i32 @llvm.ctpop.i32(i32)

declare i64 @llvm.ctpop.i64(i64)

declare <2 x i32> @llvm.ctpop.v2i32(<2 x i32>)
