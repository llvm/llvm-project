; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a35 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a57 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a53 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=cortex-a72 -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnuabi -mcpu=exynos-m1 -o - %s | FileCheck %s

%X = type { i64, i64, i64 }
declare void @f(%X*)
define void @t() {
entry:
  %tmp = alloca %X
  call void @f(%X* %tmp)
; CHECK: add x0, sp, #8
; CHECK-NEXT-NOT: mov
  call void @f(%X* %tmp)               
; CHECK: add x0, sp, #8
; CHECK-NEXT-NOT: mov
  ret void 
}
