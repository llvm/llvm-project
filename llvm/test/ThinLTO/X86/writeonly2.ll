; Check that we optimize out writeonly variables and corresponding stores.
; This test uses llvm-lto2

; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/index-const-prop.ll -o %t2.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -save-temps \
; RUN:  -r=%t2.bc,foo,pl \
; RUN:  -r=%t2.bc,bar,pl \
; RUN:  -r=%t2.bc,baz,pl \
; RUN:  -r=%t2.bc,rand, \
; RUN:  -r=%t2.bc,gBar,pl \
; RUN:  -r=%t1.bc,main,plx \
; RUN:  -r=%t1.bc,baz, \
; RUN:  -r=%t1.bc,gBar, \
; RUN:  -o %t3
; RUN: llvm-dis %t3.1.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT
; RUN: llvm-dis %t3.1.5.precodegen.bc -o - | FileCheck %s --check-prefix=CODEGEN
; Check that gFoo and gBar were eliminated from source module together
; with corresponsing stores
; RUN: llvm-dis %t3.2.5.precodegen.bc -o - | FileCheck %s --check-prefix=CODEGEN-SRC

; IMPORT:       @gBar = internal local_unnamed_addr global i32 0, align 4
; IMPORT-NEXT:  @gFoo.llvm.0 = internal unnamed_addr global i32 0, align 4
; IMPORT:       !DICompileUnit({{.*}})

; CODEGEN-NOT:  gFoo
; CODEGEN-NOT:  gBar
; CODEGEN:      i32 @main
; CODEGEN-NEXT:   %1 = tail call i32 @rand()
; CODEGEN-NEXT:   %2 = tail call i32 @rand()
; CODEGEN-NEXT:   ret i32 0

; CODEGEN-SRC-NOT:   gFoo
; CODEGEN-SRC-NOT:   gBar
; CODEGEN-SRC:       void @baz()
; CODEGEN-SRC-NEXT:    %1 = tail call i32 @rand()
; CODEGEN-SRC-NEXT:    %2 = tail call i32 @rand()
; CODEGEN-SRC-NEXT:    ret void

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; We should be able to link external definition of gBar to its declaration
@gBar = external global i32

define i32 @main() local_unnamed_addr {
  tail call void @baz()
  ret i32 0
}
declare void @baz() local_unnamed_addr
