; RUN: llvm-as < %s > %t1.bc

; Try a custom pipeline
; RUN: llvm-lto2 %t1.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,patatino,px -opt-pipeline loweratomic \
; RUN:  -aa-pipeline basic-aa
; RUN: llvm-dis < %t.o.0.4.opt.bc | FileCheck %s --check-prefix=CUSTOM

; Try the new pass manager LTO default pipeline (make sure the option
; is accepted).
; RUN: llvm-lto2 %t1.bc -o %t.o -lto-use-new-pm -r %t1.bc,patatino,px

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @patatino() {
  fence seq_cst
  ret void
}

; CUSTOM: define void @patatino() {
; CUSTOM-NEXT:   ret void
; CUSTOM-NEXT: }

; Check that invalid pipelines are caught as errors.
; RUN: not llvm-lto2 %t1.bc -o %t.o \
; RUN:  -r %t1.bc,patatino,px -opt-pipeline foogoo 2>&1 | \
; RUN:  FileCheck %s --check-prefix=ERR

; ERR: LLVM ERROR: unable to parse pass pipeline description: foogoo

; RUN: not llvm-lto2 %t1.bc -o %t.o \
; RUN:  -r %t1.bc,patatino,px -aa-pipeline patatino \
; RUN:  -opt-pipeline loweratomic 2>&1 | \
; RUN:  FileCheck %s --check-prefix=AAERR

; AAERR: LLVM ERROR: unable to parse AA pipeline description: patatino
