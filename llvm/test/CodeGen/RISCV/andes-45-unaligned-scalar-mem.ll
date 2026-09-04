; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %s | FileCheck %s

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

define void @memcpy51(ptr %dst, ptr %src) #0 {
; CHECK-LABEL: memcpy51:
; CHECK-NOT: call
; CHECK: ld
; CHECK: sd
; CHECK: ret
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %dst, ptr align 1 %src, i64 51, i1 false)
  ret void
}

attributes #0 = { "tune-cpu"="andes-45-series" }
