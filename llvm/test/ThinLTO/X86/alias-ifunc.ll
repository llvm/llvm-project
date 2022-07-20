; RUN: opt -module-summary -o %t.bc %s
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-BAR
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-BAZ
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-QUX
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-RESOLVER
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-QUUX
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-CORGE
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-GRAULT
; RUN: llvm-lto2 run %t.bc -r %t.bc,foo,px -r %t.bc,bar,px -r %t.bc,baz,px -r %t.bc,qux,px -r %t.bc,grault,px -o %t2
; RUN: llvm-nm %t2.1 | FileCheck %s --check-prefix=CHECK-SYMBOL

; CHECK-SYMBOL: i bar
; CHECK-SYMBOL: i baz
; CHECK-SYMBOL: i foo
; CHECK-SYMBOL: t foo_resolver
; CHECK-SYMBOL: i grault
; CHECK-SYMBOL: i quuz
; CHECK-SYMBOL: i qux

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@foo = ifunc i32 (i32), ptr @foo_resolver
; CHECK-RESOLVER:      (name: "foo_resolver"
; CHECK-RESOLVER-SAME: live: 1
define internal i32 (i32)* @foo_resolver() {
entry:
  ret i32 (i32)* null
}
; CHECK-BAR:      (name: "bar"
; CHECK-BAR-NOT:  summaries: (
; CHECK-BAR-SAME: ; guid = {{[0-9]+}}
@bar = alias i32 (i32), ptr @foo

; CHECK-BAZ:      (name: "baz"
; CHECK-BAZ-NOT:  summaries: (
; CHECK-BAZ-SAME: ; guid = {{[0-9]+}}
@baz = weak alias i32 (i32), ptr @foo

; CHECK-QUX:      (name: "qux"
; CHECK-QUX-NOT:  summaries: (
; CHECK-QUX-SAME: ; guid = {{[0-9]+}}
@qux = alias i32 (i32), ptr @bar

; CHECK-QUUX:      (name: "quux"
; CHECK-QUUX-SAME: live: 1
@quux = internal alias i32 (i32)* (), ptr @foo_resolver
@quuz = internal ifunc i32 (i32), ptr @quux

; CHECK-CORGE:      (name: "corge"
; CHECK-CORGE-NOT:  summaries: (
; CHECK-CORGE-SAME: ; guid = {{[0-9]+}}
@corge = internal alias i32 (i32), ptr @quuz

; CHECK-GRAULT:      (name: "grault"
; CHECK-GRAULT-NOT:  summaries: (
; CHECK-GRAULT-SAME: ; guid = {{[0-9]+}}
@grault = alias i32 (i32), ptr @corge
