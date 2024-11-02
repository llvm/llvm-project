; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/index-const-prop-alias.ll -o %t2.bc
; RUN: llvm-lto2 run %t1.bc -r=%t1.bc,main,plx -r=%t1.bc,ret_ptr,pl -r=%t1.bc,g.alias,l -r=%t1.bc,g,l \
; RUN:               %t2.bc -r=%t2.bc,g,pl -r=%t2.bc,g.alias,pl -save-temps -o %t3
; RUN: llvm-dis %t3.1.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT
; RUN: llvm-dis %t3.1.5.precodegen.bc -o - | FileCheck %s --check-prefix=CODEGEN

; When ret_ptr is preserved we return pointer to alias, so we can't internalize aliasee
; RUN: llvm-lto2 run %t1.bc -r=%t1.bc,main,plx -r=%t1.bc,ret_ptr,plx -r=%t1.bc,g.alias,l -r=%t1.bc,g,l \
; RUN:               %t2.bc -r=%t2.bc,g,pl -r=%t2.bc,g.alias,pl -save-temps -o %t4
; RUN: llvm-dis %t4.1.3.import.bc -o - | FileCheck %s --check-prefix=PRESERVED

; When g.alias is preserved we can't internalize aliasee either
; RUN: llvm-lto2 run %t1.bc -r=%t1.bc,main,plx -r=%t1.bc,ret_ptr,pl -r=%t1.bc,g.alias,l -r=%t1.bc,g,l \
; RUN:               %t2.bc -r=%t2.bc,g,pl -r=%t2.bc,g.alias,plx -save-temps -o %t5
; RUN: llvm-dis %t5.1.3.import.bc -o - | FileCheck %s --check-prefix=PRESERVED

; We currently don't support importing aliases
; IMPORT:       @g.alias = external global i32
; IMPORT-NEXT:  @g = internal global i32 42, align 4 #0
; IMPORT:  attributes #0 = { "thinlto-internalize" }

; CODEGEN:      define dso_local i32 @main
; CODEGEN-NEXT:    ret i32 42

; PRESERVED:      @g.alias = external global i32
; PRESERVED-NEXT: @g = available_externally global i32 42, align 4

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g.alias = external global i32
@g = external global i32

define i32 @main() {
  %v = load i32, ptr @g
  ret i32 %v
}

define ptr @ret_ptr() {
  ret ptr @g.alias
}
