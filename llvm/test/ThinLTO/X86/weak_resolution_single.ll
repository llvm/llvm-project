; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t2.bc %t.bc

; RUN: llvm-lto -thinlto-action=internalize %t.bc -thinlto-index=%t2.bc -exported-symbol=_foo -o - | llvm-dis -o - | FileCheck %s

; CHECK: define weak_odr void @foo()
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
define linkonce_odr void @foo() {
  ret void
}
