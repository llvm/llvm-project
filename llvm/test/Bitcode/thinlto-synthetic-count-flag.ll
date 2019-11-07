; REQUIRES: x86-registered-target
; RUN: opt -module-summary %s -o %t.o

; Ensure synthetic entry count flag is not set on distributed index
; RUN: llvm-lto2 run %t.o -o %t.out -thinlto-distributed-indexes \
; RUN:		-r %t.o,glob,plx -compute-dead=false
; RUN: llvm-bcanalyzer -dump %t.o.thinlto.bc | FileCheck %s --check-prefix=NOSYNTHETIC
; NOSYNTHETIC: <FLAGS op0=32/>

; Ensure synthetic entry count flag is set on distributed index
; when option used to enable synthetic count propagation
; RUN: llvm-lto2 run %t.o -o %t.out -thinlto-distributed-indexes \
; RUN:		-r %t.o,glob,plx -thinlto-synthesize-entry-counts \
; RUN:          -compute-dead=false
; RUN: llvm-bcanalyzer -dump %t.o.thinlto.bc | FileCheck %s --check-prefix=HASSYNTHETIC
; HASSYNTHETIC: <FLAGS op0=36/>

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@glob = global i32 0
