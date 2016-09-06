; RUN: llc -verify-machineinstrs -O0 -mcpu=pwr8 -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s
; Function Attrs: nounwind
define void @test() {
entry:
  %__a.addr.i = alloca i32, align 4
  %__b.addr.i = alloca <4 x i32>*, align 8
  %i = alloca <4 x i32>, align 16
  %j = alloca <4 x i32>, align 16
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32>* %i, align 16
  store i32 0, i32* %__a.addr.i, align 4
  store <4 x i32>* %i, <4 x i32>** %__b.addr.i, align 8
  %0 = load i32, i32* %__a.addr.i, align 4
  %1 = load <4 x i32>*, <4 x i32>** %__b.addr.i, align 8
  %2 = bitcast <4 x i32>* %1 to i8*
  %3 = getelementptr i8, i8* %2, i32 %0
  %4 = call <4 x i32> @llvm.ppc.vsx.lxvw4x(i8* %3)
; CHECK: lwa [[REG0:[0-9]+]],
; CHECK: lxvd2x [[REG1:[0-9]+]], {{[0-9]+}}, [[REG0]]
; CHECK: xxswapd [[REG1]], [[REG1]]
  store <4 x i32> %4, <4 x i32>* %j, align 16
  ret void
}

; Function Attrs: nounwind readonly
declare <4 x i32> @llvm.ppc.vsx.lxvw4x(i8*)
