; RUN: llc -mtriple=riscv64 -mattr=+zbkx -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV64ZBKX

declare i64 @llvm.riscv.xperm8.i64(i64 %a, i64 %b)

define i64 @xperm8(i64 %a, i64 %b) nounwind {
; RV64ZBKX-LABEL: xperm8:
; RV64ZBKX:       # %bb.0:
; RV64ZBKX-NEXT:    xperm8 a0, a0, a1
; RV64ZBKX-NEXT:    ret
  %tmp = call i64 @llvm.riscv.xperm8.i64(i64 %a, i64 %b)
 ret i64 %tmp
}

declare i64 @llvm.riscv.xperm4.i64(i64 %a, i64 %b)

define i64 @xperm4(i64 %a, i64 %b) nounwind {
; RV64ZBKX-LABEL: xperm4:
; RV64ZBKX:       # %bb.0:
; RV64ZBKX-NEXT:    xperm4 a0, a0, a1
; RV64ZBKX-NEXT:    ret
  %tmp = call i64 @llvm.riscv.xperm4.i64(i64 %a, i64 %b)
 ret i64 %tmp
}
