; RUN: llc -O0 -mtriple=spirv64-unknown-linux %s -o - | FileCheck %s

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
  store i8 %0, i8 addrspace(1)* @g1, align 4
  %1 = tail call i16 @llvm.ctpop.i16(i16 %x16)
  store i16 %1, i16 addrspace(1)* @g2, align 4
  %2 = tail call i32 @llvm.ctpop.i32(i32 %x32)
  store i32 %2, i32 addrspace(1)* @g3, align 4
  %3 = tail call i64 @llvm.ctpop.i64(i64 %x64)
  store i64 %3, i64 addrspace(1)* @g4, align 8
  %4 = tail call <2 x i32> @llvm.ctpop.v2i32(<2 x i32> %x2i32)
  store <2 x i32> %4, <2 x i32> addrspace(1)* @g5, align 4

  ret void
}

declare i8 @llvm.ctpop.i8(i8)

declare i16 @llvm.ctpop.i16(i16)

declare i32 @llvm.ctpop.i32(i32)

declare i64 @llvm.ctpop.i64(i64)

declare <2 x i32> @llvm.ctpop.v2i32(<2 x i32>)
