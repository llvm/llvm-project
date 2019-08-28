; RUN: opt -module-summary -o %t %s
; RUN: llvm-lto2 run %t -O0 -r %t,foo,px -o %t2
; RUN: llvm-nm %t2.1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: W foo
define linkonce_odr void @foo() {
  ret void
}
