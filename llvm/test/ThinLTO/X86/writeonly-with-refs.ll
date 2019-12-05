; RUN: opt -thinlto-bc %s -o %t1
; RUN: opt -thinlto-bc %p/Inputs/writeonly-with-refs.ll -o %t2
; RUN: llvm-lto2 run -save-temps %t1 %t2 -o %t-out \
; RUN:    -r=%t1,main,plx \
; RUN:    -r=%t1,_Z3foov,l \
; RUN:    -r=%t2,_Z3foov,pl \
; RUN:    -r=%t2,outer,pl

; @outer should have been internalized and converted to zeroinitilizer.
; RUN: llvm-dis %t-out.1.3.import.bc -o - | FileCheck %s
; RUN: llvm-dis %t-out.2.3.import.bc -o - | FileCheck %s

; CHECK: @outer = internal local_unnamed_addr global %struct.Q zeroinitializer

; Test again in distributed ThinLTO mode.
; RUN: llvm-lto2 run -save-temps %t1 %t2 -o %t-out \
; RUN:    -thinlto-distributed-indexes \
; RUN:    -r=%t1,main,plx \
; RUN:    -r=%t1,_Z3foov,l \
; RUN:    -r=%t2,_Z3foov,pl \
; RUN:    -r=%t2,outer,pl
; RUN: opt -function-import -import-all-index -enable-import-metadata -summary-file %t1.thinlto.bc %t1 -o %t1.out
; RUN: opt -function-import -import-all-index -summary-file %t2.thinlto.bc %t2 -o %t2.out
; RUN: llvm-dis %t1.out -o - | FileCheck %s
; RUN: llvm-dis %t2.out -o - | FileCheck %s

source_filename = "main.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse uwtable
define dso_local i32 @main() local_unnamed_addr {
entry:
  tail call void @_Z3foov()
  ret i32 0
}

declare dso_local void @_Z3foov() local_unnamed_addr
