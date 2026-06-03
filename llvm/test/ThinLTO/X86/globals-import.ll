; Test to ensure that we import a single copy of a global variable. This is
; important when we link in an object file twice, which is normally works when
; all symbols have either weak or internal linkage. If we import an internal
; global variable twice it will get promoted in each module, and given the same
; name as the IR hash will be identical, resulting in multiple defs when linking.
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/globals-import.ll -o %t2.bc
; RUN: opt -module-summary %p/Inputs/globals-import.ll -o %t2b.bc
; RUN: llvm-lto -thinlto-action=thinlink %t1.bc %t2.bc %t2b.bc -o %t3.index.bc

; RUN: llvm-lto -thinlto-action=import %t1.bc -thinlto-index=%t3.index.bc
; RUN: llvm-dis %t1.bc.thinlto.imported.bc -o - | FileCheck --check-prefix=IMPORT %s
; RUN: llvm-lto -thinlto-action=promote %t2.bc -thinlto-index=%t3.index.bc
; RUN: llvm-lto -thinlto-action=promote %t2b.bc -thinlto-index=%t3.index.bc
; RUN: llvm-dis %t2.bc.thinlto.promoted.bc -o - | FileCheck --check-prefix=PROMOTE1 %s
; RUN: llvm-dis %t2b.bc.thinlto.promoted.bc -o - | FileCheck --check-prefix=PROMOTE2 %s

; IMPORT: @baz.llvm.0 = internal constant i32 10, align 4

; PROMOTE1: @baz.llvm.0 = hidden constant i32 10, align 4
; PROMOTE1: define weak_odr i32 @foo() {

; Second copy of IR object should not have any symbols imported/promoted.
; PROMOTE2: @baz = internal constant i32 10, align 4
; PROMOTE2: define available_externally i32 @foo() {

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare i32 @foo()

define i32 @main() local_unnamed_addr {
  %1 = call i32 @foo()
  ret i32 %1
}
