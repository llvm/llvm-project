; RUN: llvm-as < %s > %t1.bc

; Try a custom pipeline
; RUN: llvm-lto2 %t1.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,patatino,px -opt-pipeline loweratomic
; RUN: llvm-dis < %t.o.0.4.opt.bc | FileCheck %s --check-prefix=CUSTOM

target triple = "x86_64-unknown-linux-gnu"

define void @patatino() {
  fence seq_cst
  ret void
}

; CUSTOM: define void @patatino() {
; CUSTOM-NEXT:   ret void
; CUSTOM-NEXT: }

; Check that invalid pipeline are caught as errors.
; RUN: not llvm-lto2 %t1.bc -o %t.o -save-temps \
; RUN:  -r %t1.bc,patatino,px -opt-pipeline foogoo 2>&1 | \
; RUN:  FileCheck %s --check-prefix=ERR

; ERR: LLVM ERROR: unable to parse pass pipeline description: foogoo
