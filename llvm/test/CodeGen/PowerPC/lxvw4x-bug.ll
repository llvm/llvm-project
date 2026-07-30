; RUN: llc -verify-machineinstrs -O0 -mcpu=pwr8 \
; RUN:   -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -O0 -mcpu=pwr9 \
; RUN:   -mtriple=powerpc64le-unknown-unknown < %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-P9UP --implicit-check-not xxswapd

; RUN: llc -verify-machineinstrs -O0 -mcpu=pwr9 -mattr=-power9-vector \
; RUN:   -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -O0 -mcpu=pwr10 \
; RUN:   -mtriple=powerpc64le-unknown-unknown < %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-P9UP

; RUN: llc -verify-machineinstrs -O0 -mcpu=pwr10 \
; RUN:   -mtriple=powerpc64-unknown-unknown < %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-P9UP

; Function Attrs: nounwind
define void @test() {
entry:
  %__a.addr.i = alloca i32, align 4
  %__b.addr.i = alloca ptr, align 8
  %i = alloca <4 x i32>, align 16
  %j = alloca <4 x i32>, align 16
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %i, align 16
  store i32 0, ptr %__a.addr.i, align 4
  store ptr %i, ptr %__b.addr.i, align 8
  %0 = load i32, ptr %__a.addr.i, align 4
  %1 = load ptr, ptr %__b.addr.i, align 8
  %2 = getelementptr i8, ptr %1, i32 %0
  %3 = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr %2)
; CHECK: lwa [[REG0:[0-9]+]],
; CHECK: lxvd2x [[REG1:[0-9]+]], {{[0-9]+}}, [[REG0]]
; CHECK: xxswapd [[REG1]], [[REG1]]
; CHECK-P9UP: lwa [[REG0:[0-9]+]],
; CHECK-P9UP: lxvx [[REG1:[0-9]+]], {{[0-9]+}}, [[REG0]]
  store <4 x i32> %3, ptr %j, align 16
  ret void
}

; Function Attrs: nounwind readonly
declare <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr)
