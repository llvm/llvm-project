; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t2.bc %t.bc

; RUN: llvm-lto -thinlto-action=promote %t.bc -thinlto-index=%t2.bc -exported-symbol=foo -o - | llvm-lto -thinlto-action=internalize -thinlto-module-id=%t.bc - -thinlto-index=%t2.bc -exported-symbol=foo -o - | llvm-dis -o - | FileCheck %s

; CHECK: define weak_odr void @foo()
define linkonce_odr void @foo() {
  ret void
}
