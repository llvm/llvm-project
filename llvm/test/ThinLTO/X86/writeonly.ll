; Checks that we optimize writeonly variables and corresponding stores using llvm-lto
; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/index-const-prop.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.index.bc %t1.bc %t2.bc

; Check that we optimize write-only variables
; RUN: llvm-lto -thinlto-action=import -exported-symbol=main  %t1.bc -thinlto-index=%t3.index.bc -o %t1.imported.bc -stats 2>&1 | FileCheck %s --check-prefix=STATS
; RUN: llvm-dis %t1.imported.bc -o - | FileCheck %s --check-prefix=IMPORT
; RUN: llvm-lto -thinlto-action=optimize %t1.imported.bc -o - | llvm-dis - -o - | FileCheck %s --check-prefix=OPTIMIZE

; IMPORT:      @gBar = internal local_unnamed_addr global i32 0, align 4, !dbg !0
; IMPORT-NEXT: @gFoo.llvm.0 = internal unnamed_addr global i32 0, align 4, !dbg !5
; IMPORT: !DICompileUnit({{.*}})

; STATS:  2 module-summary-index - Number of live global variables marked write only 

; Check that we've optimized out variables and corresponding stores
; OPTIMIZE-NOT:  gFoo
; OPTIMIZE-NOT:  gBar
; OPTIMIZE:      i32 @main
; OPTIMIZE-NEXT:   %1 = tail call i32 @rand()
; OPTIMIZE-NEXT:   %2 = tail call i32 @rand()
; OPTIMIZE-NEXT:   ret i32 0

; Confirm that with -propagate-attrs=false we no longer do write-only importing
; RUN: llvm-lto -propagate-attrs=false -thinlto-action=import -exported-symbol=main  %t1.bc -thinlto-index=%t3.index.bc -o %t1.imported.bc -stats 2>&1 | FileCheck %s --check-prefix=STATS-NOPROP
; RUN: llvm-dis %t1.imported.bc -o - | FileCheck %s --check-prefix=IMPORT-NOPROP
; STATS-NOPROP-NOT: Number of live global variables marked write only
; IMPORT-NOPROP:      @gBar = available_externally
; IMPORT-NOPROP-NEXT: @gFoo.llvm.0 = available_externally

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@gBar = external global i32

; Should not be counted in the stats of live write only variables since it is
; dead and will be dropped anyway.
@gDead = internal unnamed_addr global i32 1, align 4

define i32 @main() local_unnamed_addr {
  tail call void @baz()
  ret i32 0
}
declare void @baz() local_unnamed_addr
