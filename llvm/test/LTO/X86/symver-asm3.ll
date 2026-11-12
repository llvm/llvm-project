; Test special handling of @@@.

; RUN: llvm-as < %s >%t1
; RUN: llvm-nm %t1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm "foo1:"
; CHECK-DAG: t foo1

module asm ".symver foo1, foo@@@VER1"
; CHECK-DAG: t foo@@VER1

module asm ".global foo2"
module asm ".symver foo2, foo@@@VER2"
; CHECK-DAG: U foo2
; CHECK-DAG: U foo@VER2
module asm "call foo2"

module asm ".symver foo3, foo@@@VER3"
; CHECK-DAG: t foo@@VER3

module asm ".symver foo4, foo@@@VER4"
; CHECK-DAG: T foo@@VER4

module asm ".symver foo5, foo@@@VER5"
; CHECK-DAG: U foo@VER5

module asm "foo3:"
; CHECK-DAG: t foo3

module asm ".local foo1"
module asm ".local foo3"

define void @foo4() {
; CHECK-DAG: T foo4
  ret void
}

declare void @foo5()
; CHECK-DAG: U foo5
