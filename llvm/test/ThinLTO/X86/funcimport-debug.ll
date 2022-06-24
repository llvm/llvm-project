; Test of function importing debug messages

; Require asserts for -debug-only
; REQUIRES: asserts

; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/funcimport-debug.ll -o %t2.bc

; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -r=%t1.bc,_main,plx \
; RUN:     -r=%t1.bc,_foo,l \
; RUN:     -r=%t1.bc,_baz,l \
; RUN:     -r=%t2.bc,_foo,plx \
; RUN:     -r=%t2.bc,_baz,plx \
; RUN:     -thinlto-threads=1 \
; RUN:     -debug-only=function-import 2>&1 | FileCheck %s

; CHECK-DAG: Initialize import for 6699318081062747564 (foo)
; CHECK-DAG: Initialize import for 15822663052811949562 (main)
; CHECK-DAG: ref -> 7546896869197086323 (baz)
; CHECK-DAG: edge -> 6699318081062747564 (foo) Threshold:100

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define i32 @main() #0 {
entry:
  call void (...) @foo()
  %0 = load i32, i32* @baz, align 4
  ret i32 %0
}

declare void @foo(...) #1
@baz = external global i32
