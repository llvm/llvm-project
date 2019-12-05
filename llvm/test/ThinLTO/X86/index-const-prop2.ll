; Check constant propagation in thinlto combined summary. This allows us to do 2 things:
;  1. Internalize global definition which is not used externally if all accesses to it are read-only
;  2. Make a local copy of internal definition if all accesses to it are readonly. This allows constant
;     folding it during optimziation phase.
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/index-const-prop.ll -o %t2.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -save-temps \
; RUN:  -r=%t2.bc,foo,pl \
; RUN:  -r=%t2.bc,bar,pl \
; RUN:  -r=%t2.bc,baz,pl \
; RUN:  -r=%t2.bc,rand, \
; RUN:  -r=%t2.bc,gBar,pl \
; RUN:  -r=%t1.bc,main,plx \
; RUN:  -r=%t1.bc,main2,pl \
; RUN:  -r=%t1.bc,foo, \
; RUN:  -r=%t1.bc,bar, \
; RUN:  -r=%t1.bc,baz, \
; RUN:  -r=%t1.bc,gBar, \
; RUN:  -o %t3
; RUN: llvm-dis %t3.1.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT
; RUN: llvm-dis %t3.1.5.precodegen.bc -o - | FileCheck %s --check-prefix=CODEGEN

; Now check that we won't internalize global (gBar) if it's externally referenced
; RUN: llvm-lto2 run %t1.bc %t2.bc -save-temps \
; RUN:  -r=%t2.bc,foo,pl \
; RUN:  -r=%t2.bc,bar,pl \
; RUN:  -r=%t2.bc,baz,pl \
; RUN:  -r=%t2.bc,rand, \
; RUN:  -r=%t2.bc,gBar,plx \
; RUN:  -r=%t1.bc,main,plx \
; RUN:  -r=%t1.bc,main2,pl \
; RUN:  -r=%t1.bc,foo, \
; RUN:  -r=%t1.bc,bar, \
; RUN:  -r=%t1.bc,baz, \
; RUN:  -r=%t1.bc,gBar, \
; RUN:  -o %t4
; RUN: llvm-dis %t4.1.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT2

; Run again but with main2 exported instead of main to check that write only
; variables are optimized out.
; RUN: llvm-lto2 run %t1.bc %t2.bc -save-temps \
; RUN:  -r=%t2.bc,foo,pl \
; RUN:  -r=%t2.bc,bar,pl \
; RUN:  -r=%t2.bc,baz,pl \
; RUN:  -r=%t2.bc,rand, \
; RUN:  -r=%t2.bc,gBar,pl \
; RUN:  -r=%t1.bc,main,pl \
; RUN:  -r=%t1.bc,main2,plx \
; RUN:  -r=%t1.bc,foo, \
; RUN:  -r=%t1.bc,bar, \
; RUN:  -r=%t1.bc,baz, \
; RUN:  -r=%t1.bc,gBar, \
; RUN:  -o %t5
; RUN: llvm-dis %t5.1.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT-WRITEONLY
; RUN: llvm-dis %t5.1.5.precodegen.bc -o - | FileCheck %s --check-prefix=CODEGEN2
; Check that gFoo and gBar were eliminated from source module together
; with corresponsing stores
; RUN: llvm-dis %t5.2.5.precodegen.bc -o - | FileCheck %s --check-prefix=CODEGEN2-SRC

; IMPORT:       @gFoo.llvm.0 = internal unnamed_addr global i32 1, align 4
; IMPORT-NEXT:  @gBar = internal local_unnamed_addr global i32 2, align 4
; IMPORT:       !DICompileUnit({{.*}})

; Write only variables are imported with a zero initializer.
; IMPORT-WRITEONLY:  @gFoo.llvm.0 = internal unnamed_addr global i32 0
; IMPORT-WRITEONLY:  @gBar = internal local_unnamed_addr global i32 0

; CODEGEN:        i32 @main()
; CODEGEN-NEXT:     ret i32 3

; IMPORT2: @gBar = available_externally dso_local local_unnamed_addr global i32 2, align 4

; CODEGEN2:      i32 @main2
; CODEGEN2-NEXT:   %1 = tail call i32 @rand()
; CODEGEN2-NEXT:   %2 = tail call i32 @rand()
; CODEGEN2-NEXT:   ret i32 0

; CODEGEN2-SRC:       void @baz()
; CODEGEN2-SRC-NEXT:    %1 = tail call i32 @rand()
; CODEGEN2-SRC-NEXT:    %2 = tail call i32 @rand()
; CODEGEN2-SRC-NEXT:    ret void

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; We should be able to link external definition of gBar to its declaration
@gBar = external global i32

define i32 @main() local_unnamed_addr {
  %call = tail call i32 bitcast (i32 (...)* @foo to i32 ()*)()
  %call1 = tail call i32 bitcast (i32 (...)* @bar to i32 ()*)()
  %add = add nsw i32 %call1, %call
  ret i32 %add
}

define i32 @main2() local_unnamed_addr {
  tail call void @baz()
  ret i32 0
}

declare i32 @foo(...) local_unnamed_addr

declare i32 @bar(...) local_unnamed_addr

declare void @baz() local_unnamed_addr
