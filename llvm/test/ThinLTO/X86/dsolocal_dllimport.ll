; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %S/Inputs/dsolocal_dllimport.ll -o %t2.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t -r=%t1.bc,bar,px -r=%t1.bc,__imp_foo, -r=%t2.bc,foo -save-temps
; RUN: llvm-dis < %t.1.3.import.bc | FileCheck %s

; If a user (dllimport) is LTOed with a library, check that we replace dllimport with dso_local.

; CHECK: declare dso_local void @foo()

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.0"
define void @bar() {
  call void @foo()
  ret void
}
declare dllimport void @foo()
