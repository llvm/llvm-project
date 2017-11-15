; RUN: llvm-as -o %t %s
; RUN: llvm-lto2 run %t -r %t,foo, -r %t,baz,p -o %t2 -save-temps
; RUN: llvm-dis -o - %t2.0.0.preopt.bc | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--fuchsia"

; CHECK: declare void @foo()
@foo = weak alias void(), void()* @bar

define internal void @bar() {
  ret void
}

define void()* @baz() {
  ret void()* @foo
}
