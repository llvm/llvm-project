; Check that we internalize external constant if it is accessed
; purely by non-volatile loads.
; RUN: opt -thinlto-bc %s -o %t-main
; RUN: opt -thinlto-bc %p/Inputs/import-ro-constant-foo.ll -o %t-foo
; RUN: opt -thinlto-bc %p/Inputs/import-ro-constant-bar.ll -o %t-bar
; RUN: llvm-lto2 run -save-temps -o %t-out %t-main %t-foo %t-bar \
; RUN:      -r=%t-foo,foo,pl \
; RUN:      -r=%t-main,main,plx \
; RUN:      -r=%t-main,_Z3barv,l \
; RUN:      -r=%t-main,foo, \
; RUN:      -r=%t-bar,_Z3barv,pl \
; RUN:      -r=%t-bar,foo,
; RUN: llvm-dis %t-out.1.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT
; RUN: llvm-dis %t-out.1.4.opt.bc -o - | FileCheck %s --check-prefix=OPT

; IMPORT: @foo = internal local_unnamed_addr constant i32 21, align 4 #0
; IMPORT: attributes #0 = { "thinlto-internalize" }
; OPT:      i32 @main()
; OPT-NEXT: entry:
; OPT-NEXT:   ret i32 42

source_filename = "main.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = external dso_local local_unnamed_addr constant i32, align 4
define dso_local i32 @main() local_unnamed_addr {
entry:
  %0 = load i32, ptr @foo, align 4
  %call = tail call i32 @_Z3barv()
  %add = add nsw i32 %call, %0
  ret i32 %add
}
declare dso_local i32 @_Z3barv() local_unnamed_addr
