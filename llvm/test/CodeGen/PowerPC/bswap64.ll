; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64le-- -mcpu=pwr9 | FileCheck %s

declare i64 @llvm.bswap.i64(i64)

; CHECK: mtvsrdd
; CHECK: xxbrd
; CHECK: mfvsrd
define i64 @bswap64(i64 %x) {
entry:
  %0 = call i64 @llvm.bswap.i64(i64 %x)
  ret i64 %0
}

