; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/globals-import-cf-baz.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink %t1.bc %t2.bc -o %t3.index.bc

; RUN: llvm-lto -thinlto-action=import -exported-symbol=main %t1.bc -thinlto-index=%t3.index.bc
; RUN: llvm-dis %t1.bc.thinlto.imported.bc -o - | FileCheck --check-prefix=IMPORT %s
; RUN: llvm-lto -thinlto-action=optimize %t1.bc.thinlto.imported.bc -o %t1.bc.thinlto.opt.bc
; RUN: llvm-dis %t1.bc.thinlto.opt.bc -o - | FileCheck --check-prefix=OPTIMIZE %s

; IMPORT: @baz = internal local_unnamed_addr constant i32 10

; OPTIMIZE:       define noundef i32 @main()
; OPTIMIZE-NEXT:    ret i32 10

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@baz = external local_unnamed_addr constant i32, align 4

define i32 @main() local_unnamed_addr {
  %1 = load i32, ptr @baz, align 4
  ret i32 %1
}
