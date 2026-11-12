; REQUIRES: x86-registered-target
; RUN: opt -module-summary %s -o %t.o

;; By default, the indexing step should perform and set the appropriate index
;; flags for dead stripping, attribute propagation, DSO local propagation,
;; and internalization/promotion.
; RUN: llvm-lto2 run %t.o -o %t.out -thinlto-distributed-indexes \
; RUN:		-r %t.o,glob,plx
; RUN: llvm-bcanalyzer -dump %t.o.thinlto.bc | FileCheck %s --check-prefix=ALL
;; The flag value should be 0x461 aka 1121:
;; 0x1: Dead stripping
;; 0x20: Attribute propagation
;; 0x40: DSO local propagation
;; 0x400: Internalization/promotion
; ALL: <FLAGS op0=1121/>

;; Ensure dead stripping performed flag is not set on distributed index
;; when option used to disable dead stripping computation.
; RUN: llvm-lto2 run %t.o -o %t.out -thinlto-distributed-indexes \
; RUN:		-r %t.o,glob,plx -compute-dead=false
; RUN: llvm-bcanalyzer -dump %t.o.thinlto.bc | FileCheck %s --check-prefix=NODEAD
;; Flag should be 0x460 aka 1120.
; NODEAD: <FLAGS op0=1120/>

;; Disabling attribute propagation should disable that as well as DSO local
;; propagation.
; RUN: llvm-lto2 run %t.o -o %t.out -thinlto-distributed-indexes \
; RUN:		-r %t.o,glob,plx -propagate-attrs=false
; RUN: llvm-bcanalyzer -dump %t.o.thinlto.bc | FileCheck %s --check-prefix=NOPROP
;; Flag should be 0x401 aka 1025.
; NOPROP: <FLAGS op0=1025/>

;; Note there isn't currently a way to disable internalization+promotion, which
;; are performed together.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@glob = global i32 0
