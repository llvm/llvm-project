; RUN: opt -module-summary -o %t %s
; RUN: opt -module-summary -o %t2 %S/Inputs/dead-strip-alias.ll
; RUN: llvm-lto2 run %t -r %t,main,px -r %t,alias,p -r %t,external, \
; RUN:               %t2 -r %t2,external,p \
; RUN: -save-temps -o %t3
; RUN: llvm-nm %t3.2 | FileCheck %s

; CHECK: D external

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@alias = alias i8*, i8** @internal

@internal = internal global i8* @external
@external = external global i8

define i8** @main() {
  ret i8** @alias
}
