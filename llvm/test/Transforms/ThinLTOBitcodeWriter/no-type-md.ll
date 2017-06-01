; RUN: opt -thinlto-bc -o %t %s
; RUN: llvm-dis -o - %t | FileCheck %s
; RUN: llvm-bcanalyzer -dump %t | FileCheck --check-prefix=BCA %s

; BCA: <GLOBALVAL_SUMMARY_BLOCK

; CHECK: @g = global i8 42
@g = global i8 42

; CHECK: define void @f()
define void @f() {
  ret void
}
