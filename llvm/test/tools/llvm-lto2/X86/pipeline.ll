; RUN: llvm-as < %s > %t1.bc

; Try the default pipeline and check is BasicAA is invoked.
; RUN: llvm-lto2 run %t1.bc -o %t.o -r %t1.bc,patatino,px -debug-pass-manager \
; RUN:  2>&1 | FileCheck %s --check-prefix=DEFAULT
; DEFAULT: Running analysis: BasicAA on patatino

; Try a custom pipeline
; RUN: llvm-lto2 run %t1.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,patatino,px -opt-pipeline lower-atomic \
; RUN:  -aa-pipeline basic-aa
; RUN: llvm-dis < %t.o.0.4.opt.bc | FileCheck %s --check-prefix=CUSTOM

; Try the new pass manager LTO default pipeline (make sure the option
; is accepted).
; RUN: llvm-lto2 run %t1.bc -o %t.o -r %t1.bc,patatino,px

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @patatino() {
  fence seq_cst
  ret void
}

; CUSTOM: define void @patatino() {
; CUSTOM-NEXT:   ret void
; CUSTOM-NEXT: }

; Check that invalid pipelines are caught as errors.
; RUN: not llvm-lto2 run %t1.bc -o %t.o \
; RUN:  -r %t1.bc,patatino,px -opt-pipeline foogoo 2>&1 | \
; RUN:  FileCheck %s --check-prefix=ERR

; ERR: LLVM ERROR: unable to parse pass pipeline description 'foogoo': unknown pass name 'foogoo'

; RUN: not llvm-lto2 run %t1.bc -o %t.o \
; RUN:  -r %t1.bc,patatino,px -aa-pipeline patatino \
; RUN:  -opt-pipeline lower-atomic 2>&1 | \
; RUN:  FileCheck %s --check-prefix=AAERR

; AAERR: LLVM ERROR: unable to parse AA pipeline description 'patatino': unknown alias analysis name 'patatino'
