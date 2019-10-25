; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-lto2 run -save-temps %t.bc -o %t.out \
; RUN:    -r=%t.bc,foo,plx \
; RUN:    -r=%t.bc,bar,lx

; Check that we don't internalize `bar` during promotion,
; because foo and bar are members of the same comdat
; RUN: llvm-dis %t.out.1.1.promote.bc -o - | FileCheck %s

; Thin LTO internalization shouldn't internalize `bar` as well
; RUN: llvm-dis %t.out.1.2.internalize.bc -o - | FileCheck %s

; CHECK: define linkonce_odr dso_local i32 @bar() comdat($foo)

$foo = comdat any

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local i32 @bar() comdat($foo) {
entry:
  ret i32 33
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local i32 @foo() comdat {
entry:
  %call = call i32 @bar()
  %add = add nsw i32 42, %call
  ret i32 %add
}
