; RUN: llc -mtriple=riscv32 -mattr=+zbkx -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV32ZBKX

declare i32 @llvm.riscv.xperm8.i32(i32 %a, i32 %b)

define i32 @xperm8(i32 %a, i32 %b) nounwind {
; RV32ZBKX-LABEL: xperm8:
; RV32ZBKX:       # %bb.0:
; RV32ZBKX-NEXT:    xperm8 a0, a0, a1
; RV32ZBKX-NEXT:    ret
  %tmp = call i32 @llvm.riscv.xperm8.i32(i32 %a, i32 %b)
 ret i32 %tmp
}

declare i32 @llvm.riscv.xperm4.i32(i32 %a, i32 %b)

define i32 @xperm4(i32 %a, i32 %b) nounwind {
; RV32ZBKX-LABEL: xperm4:
; RV32ZBKX:       # %bb.0:
; RV32ZBKX-NEXT:    xperm4 a0, a0, a1
; RV32ZBKX-NEXT:    ret
  %tmp = call i32 @llvm.riscv.xperm4.i32(i32 %a, i32 %b)
 ret i32 %tmp
}
