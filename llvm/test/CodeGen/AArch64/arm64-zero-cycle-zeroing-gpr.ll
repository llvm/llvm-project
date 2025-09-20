; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s -check-prefixes=ALL,NOZCZ-GPR
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+zcz-gpr32 | FileCheck %s -check-prefixes=ALL,ZCZ-GPR32
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+zcz-gpr64 | FileCheck %s -check-prefixes=ALL,ZCZ-GPR64
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=generic | FileCheck %s -check-prefixes=ALL,NOZCZ-GPR
; RUN: llc < %s -mtriple=arm64-apple-ios -mcpu=cyclone | FileCheck %s -check-prefixes=ALL,ZCZ-GPR32,ZCZ-GPR64
; RUN: llc < %s -mtriple=arm64-apple-macosx -mcpu=apple-m1 | FileCheck %s -check-prefixes=ALL,ZCZ-GPR32,ZCZ-GPR64
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mcpu=exynos-m3 | FileCheck %s -check-prefixes=ALL,NOZCZ-GPR
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mcpu=kryo | FileCheck %s -check-prefixes=ALL,ZCZ-GPR32,ZCZ-GPR64
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mcpu=falkor | FileCheck %s -check-prefixes=ALL,ZCZ-GPR32,ZCZ-GPR64

define i8 @ti8() {
entry:
; ALL-LABEL: ti8:
; NOZCZ-GPR: mov w0, wzr
; ZCZ-GPR32: mov w0, #0
  ret i8 0
}

define i16 @ti16() {
entry:
; ALL-LABEL: ti16:
; NOZCZ-GPR: mov w0, wzr
; ZCZ-GPR32: mov w0, #0
  ret i16 0
}

define i32 @ti32() {
entry:
; ALL-LABEL: ti32:
; NOZCZ-GPR: mov w0, wzr
; ZCZ-GPR32: mov w0, #0
  ret i32 0
}

define i64 @ti64() {
entry:
; ALL-LABEL: ti64:
; NOZCZ-GPR: mov x0, xzr
; ZCZ-GPR64: mov x0, #0
  ret i64 0
}
