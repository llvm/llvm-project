; Check constant propagation in thinlto combined summary. This allows us to do 2 things:
;  1. Internalize global definition which is not used externally if all accesses to it are read-only
;  2. Make a local copy of internal definition if all accesses to it are readonly. This allows constant
;     folding it during optimziation phase.

; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/index-const-prop.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.index.bc %t1.bc %t2.bc
; RUN: llvm-lto -thinlto-action=import -exported-symbol=main  %t1.bc -thinlto-index=%t3.index.bc -o %t1.imported.bc -stats 2>&1 | FileCheck %s --check-prefix=STATS
; RUN: llvm-dis %t1.imported.bc -o - | FileCheck %s --check-prefix=IMPORT
; RUN: llvm-lto -thinlto-action=optimize %t1.imported.bc -o - | llvm-dis - -o - | FileCheck %s --check-prefix=OPTIMIZE

; STATS: 2 module-summary-index - Number of live global variables marked read only

; Check that we don't internalize gBar when it is exported
; RUN: llvm-lto -thinlto-action=import -exported-symbol main -exported-symbol gBar  %t1.bc -thinlto-index=%t3.index.bc -o %t1.imported2.bc
; RUN: llvm-dis %t1.imported2.bc -o - | FileCheck %s --check-prefix=IMPORT2

; IMPORT:      @gBar = internal local_unnamed_addr global i32 2, align 4, !dbg !0
; IMPORT-NEXT: @gFoo.llvm.0 = internal unnamed_addr global i32 1, align 4, !dbg !5
; IMPORT: !DICompileUnit({{.*}})

; OPTIMIZE:        define i32 @main
; OPTIMIZE-NEXT:     ret i32 3

; IMPORT2: @gBar = available_externally local_unnamed_addr global i32 2, align 4, !dbg !0

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@gBar = external global i32

; Should not be counted in the stats of live read only variables since it is
; dead and will be dropped anyway.
@gDead = internal unnamed_addr global i32 1, align 4

define i32 @main() local_unnamed_addr {
  %call = tail call i32 bitcast (i32 (...)* @foo to i32 ()*)()
  %call1 = tail call i32 bitcast (i32 (...)* @bar to i32 ()*)()
  %add = add nsw i32 %call1, %call
  ret i32 %add
}

declare i32 @foo(...) local_unnamed_addr

declare i32 @bar(...) local_unnamed_addr
