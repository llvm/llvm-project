; RUN: opt -thinlto-bc %s -o %t1.bc
; RUN: opt -thinlto-bc %p/Inputs/import-symver-foo.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink %t1.bc %t2.bc -o %t3.index.bc

; RUN: llvm-lto -thinlto-action=import -exported-symbol=main %t1.bc -thinlto-index=%t3.index.bc
; RUN: llvm-dis %t1.bc.thinlto.imported.bc -o - | FileCheck --check-prefix=IMPORT %s

; RUN: llvm-lto -thinlto-action=import -exported-symbol=main -import-instr-limit=0 %t1.bc -thinlto-index=%t3.index.bc
; RUN: llvm-dis %t1.bc.thinlto.imported.bc -o - | FileCheck --check-prefix=NOIMPORT %s

; When @bar gets imported, the symver must be imported too.
; IMPORT: module asm ".symver bar, bar@BAR_1.2.3"
; IMPORT: declare dso_local i32 @bar()

; When @bar isn't imported, the symver is also not imported.
; NOIMPORT-NOT: module asm
; NOIMPORT-NOT: declare dso_local i32 @bar()

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local i32 @foo()

define dso_local i32 @main() {
entry:
  %call = tail call i32 @foo()
  ret i32 %call
}
