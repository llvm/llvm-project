; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/index-const-prop-gvref.ll -o %t2.bc
; RUN: llvm-lto2 run -relocation-model=static -save-temps %t2.bc -r=%t2.bc,b,pl -r=%t2.bc,a,pl \
; RUN:   %t1.bc -r=%t1.bc,main,plx -r=%t1.bc,a, -r=%t1.bc,b, -o %t3
; RUN: llvm-dis %t3.1.3.import.bc -o - | FileCheck %s --check-prefix=SRC
; RUN: llvm-dis %t3.2.3.import.bc -o - | FileCheck %s --check-prefix=DEST

;; When producing an ELF DSO, clear dso_local for declarations to avoid direct access.
; RUN: llvm-lto2 run -relocation-model=pic -save-temps %t2.bc -r=%t2.bc,b,pl -r=%t2.bc,a,pl \
; RUN:   %t1.bc -r=%t1.bc,main,plx -r=%t1.bc,a, -r=%t1.bc,b, -o %t4
; RUN: llvm-dis %t4.1.3.import.bc -o - | FileCheck %s --check-prefix=SRC
; RUN: llvm-dis %t4.2.3.import.bc -o - | FileCheck %s --check-prefix=DEST_DSO

; No variable in the source module should have been internalized
; SRC:      @b = dso_local global i32* @a
; SRC-NEXT: @a = dso_local global i32 42

; We can't internalize globals referenced by other live globals
; DEST:          @b = external dso_local global i32*
; DEST-NEXT:     @a = available_externally dso_local global i32 42, align 4
; DEST_DSO:      @b = external global i32*
; DEST_DSO-NEXT: @a = available_externally global i32 42, align 4

;; Test old API.
;; When producing an ELF DSO, clear dso_local for declarations to avoid direct access.
; RUN: llvm-lto -thinlto-action=run %t2.bc %t1.bc -relocation-model=static -thinlto-save-temps=%t5.
; RUN: llvm-dis < %t5.0.3.imported.bc | FileCheck %s --check-prefix=OLDAPI_SRC
; RUN: llvm-dis < %t5.1.3.imported.bc | FileCheck %s --check-prefix=OLDAPI_DST
; RUN: llvm-lto -thinlto-action=run %t2.bc %t1.bc -relocation-model=pic -thinlto-save-temps=%t6.
; RUN: llvm-dis < %t6.0.3.imported.bc | FileCheck %s --check-prefix=OLDAPI_SRC
; RUN: llvm-dis < %t6.1.3.imported.bc | FileCheck %s --check-prefix=OLDAPI_DST_DSO

; OLDAPI_SRC:      @b = internal global i32* @a, align 8
; OLDAPI_SRC-NEXT: @a = dso_local global i32 42, align 4
; OLDAPI_DST:      @b = external dso_local global i32*
; OLDAPI_DST-NEXT: @a = available_externally dso_local global i32 42, align 4
; OLDAPI_DST_DSO:      @b = external global i32*
; OLDAPI_DST_DSO-NEXT: @a = available_externally global i32 42, align 4

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external global i32
@b = external global i32*

define i32 @main() {
  %p = load i32*, i32** @b, align 8  
  store i32 33, i32* %p, align 4
  %v = load i32, i32* @a, align 4
  ret i32 %v
}
