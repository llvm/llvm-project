; RUN: llc -O0 -mtriple=spirv64-unknown-linux %s -o - | FileCheck %s

; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] s_abs
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] s_abs

@ga = addrspace(1) global i32 undef, align 4
@gb = addrspace(1) global <4 x i32> undef, align 4

define dso_local spir_kernel void @test(i32 %a, <4 x i32> %b) local_unnamed_addr {
entry:
  %0 = tail call i32 @llvm.abs.i32(i32 %a, i1 0)
  store i32 %0, i32 addrspace(1)* @ga, align 4
  %1 = tail call <4 x i32> @llvm.abs.v4i32(<4 x i32> %b, i1 0)
  store <4 x i32> %1, <4 x i32> addrspace(1)* @gb, align 4

  ret void
}

declare i32 @llvm.abs.i32(i32, i1)

declare <4 x i32> @llvm.abs.v4i32(<4 x i32>, i1)
