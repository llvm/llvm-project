; RUN: llc < %s -mtriple=mips -mcpu=mips2 | FileCheck %s -check-prefixes=ALL,GP32
; RUN: llc < %s -mtriple=mips -mcpu=mips32 | FileCheck %s -check-prefixes=ALL,GP32
; RUN: llc < %s -mtriple=mips -mcpu=mips32r2 | FileCheck %s -check-prefixes=ALL,GP32
; RUN: llc < %s -mtriple=mips -mcpu=mips32r3 | FileCheck %s -check-prefixes=ALL,GP32
; RUN: llc < %s -mtriple=mips -mcpu=mips32r5 | FileCheck %s -check-prefixes=ALL,GP32
; RUN: llc < %s -mtriple=mips -mcpu=mips32r6 | FileCheck %s -check-prefixes=ALL,GP32
; RUN: llc < %s -mtriple=mips64 -mcpu=mips3 | FileCheck %s -check-prefix=ALL
; RUN: llc < %s -mtriple=mips64 -mcpu=mips4 | FileCheck %s -check-prefix=ALL
; RUN: llc < %s -mtriple=mips64 -mcpu=mips64 | FileCheck %s -check-prefix=ALL
; RUN: llc < %s -mtriple=mips64 -mcpu=mips64r2 | FileCheck %s -check-prefix=ALL
; RUN: llc < %s -mtriple=mips64 -mcpu=mips64r3 | FileCheck %s -check-prefix=ALL
; RUN: llc < %s -mtriple=mips64 -mcpu=mips64r5 | FileCheck %s -check-prefix=ALL
; RUN: llc < %s -mtriple=mips64 -mcpu=mips64r6 | FileCheck %s -check-prefix=ALL

define i64 @add_i64(i64 %a) {
  ; GP32-LABEL: add_i64

  ; GP32-NOT:    move $[[T0:[0-9]+]], $[[T0]]
  %r = add i64 5, %a
  ret i64 %r
}

define i128 @add_i128(i128 %a) {
  ; ALL-LABEL: add_i128

  ; ALL-NOT:    move $[[T0:[0-9]+]], $[[T0]]
  %r = add i128 5, %a
  ret i128 %r
}
