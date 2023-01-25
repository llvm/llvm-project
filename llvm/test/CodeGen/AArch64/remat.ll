; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a34 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a35 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a53 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a55 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a57 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a65 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a65ae -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a72 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a73 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a75 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a77 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a78 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-x1 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=neoverse-e1 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=neoverse-n1 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=neoverse-n2 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=exynos-m3 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=exynos-m4 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=exynos-m5 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=falkor -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=saphira -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=kryo -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=thunderx2t99 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=tsv110 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mattr=+custom-cheap-as-move -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=thunderx3t110 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=ampere1 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=ampere1a -o - %s | FileCheck %s

%X = type { i64, i64, i64 }
declare void @f(ptr)
define void @t() {
entry:
  %tmp = alloca %X
  call void @f(ptr %tmp)
; CHECK: add x0, sp, #8
; CHECK-NOT: mov
; CHECK-NEXT: bl f
  call void @f(ptr %tmp)
; CHECK: add x0, sp, #8
; CHECK-NOT: mov
; CHECK-NEXT: bl f
  ret void
}
