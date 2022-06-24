; RUN: split-file %s %t.dir

; RUN: opt -module-summary %t.dir/1.ll -o %t1.bc
; RUN: opt -module-summary %t.dir/2.ll -o %t2.bc

; RUN: llvm-lto2 run -save-temps %t1.bc %t2.bc -o %t.out \
; RUN:               -r=%t1.bc,main,plx \
; RUN:               -r=%t1.bc,G \
; RUN:               -r=%t2.bc,G,pl
; RUN: llvm-dis %t.out.1.3.import.bc -o -  | FileCheck %s
; RUN: llvm-dis %t.out.2.3.import.bc -o -  | FileCheck %s

; Test that a non-prevailing def with interposable linkage doesn't prevent
; importing a suitable definition from a prevailing module.

; CHECK: @G = internal local_unnamed_addr global i32 42

;--- 1.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@G = weak dso_local local_unnamed_addr global i32 0, align 4

define dso_local i32 @main() local_unnamed_addr {
  %1 = load i32, i32* @G, align 4
  ret i32 %1
}

;--- 2.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@G = dso_local local_unnamed_addr global i32 42, align 4
