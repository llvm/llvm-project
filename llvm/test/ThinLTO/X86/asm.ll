; RUN: split-file %s %t
; RUN: opt -module-summary %t/a.s -o %t/a.bc
; RUN: opt -module-summary %t/b.s -o %t/b.bc
; RUN: llvm-nm %t/a.bc | FileCheck %s --check-prefix=NM

; RUN: llvm-lto2 run %t/a.bc %t/b.bc -o %t/out -save-temps -r=%t/a.bc,ref,plx -r=%t/b.bc,ff_h264_cabac_tables,pl
; RUN: llvm-dis < %t/out.2.2.internalize.bc | FileCheck %s

;--- a.s
;; IR symtab does not track inline asm symbols, so we don't know
;; ff_h264_cabac_tables is undefined.
; NM-NOT: {{.}}
; NM:     ---------------- T ref
; NM-NOT: {{.}}
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i8* @ref() {
entry:
  %0 = tail call i8* asm sideeffect "lea ff_h264_cabac_tables(%rip), $0", "=&r,~{dirflag},~{fpsr},~{flags}"()
  ret i8* %0
}

;--- b.s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; ff_h264_cabac_tables has __attribute__((used)) in the source code, which means
;; its definition must be retained because there can be references the compiler
;; cannot see (inline asm reference). Test we don't internalize it.
; CHECK: @ff_h264_cabac_tables = dso_local constant [1 x i8] c"\09"
@ff_h264_cabac_tables = dso_local constant [1 x i8] c"\09"

@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast ([1 x i8]* @ff_h264_cabac_tables to i8*)], section "llvm.metadata"
