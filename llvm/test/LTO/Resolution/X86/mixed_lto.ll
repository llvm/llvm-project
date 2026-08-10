; Test mixed-mode LTO (mix of regular and thin LTO objects)
; RUN: opt %s -o %t1.o
; RUN: opt -module-summary %p/Inputs/mixed_lto.ll -o %t2.o

; RUN: llvm-lto2 run -o %t3.o %t2.o %t1.o -r %t2.o,main,px -r %t2.o,g, -r %t1.o,g,px

; Task 0 is the regular LTO file (this file)
; RUN: llvm-nm %t3.o.0 | FileCheck %s --check-prefix=NM0
; NM0: T g

; Task 1 is the (first) ThinLTO file (Inputs/mixed_lto.ll)
; RUN: llvm-nm %t3.o.1 | FileCheck %s --check-prefix=NM1
; NM1-DAG: T main
; NM1-DAG: U g

; Do the same test again, but with the regular and thin LTO modules in the same file.
; RUN: llvm-cat -b -o %t4.o %t2.o %t1.o
; RUN: llvm-lto2 run -o %t5.o %t4.o -r %t4.o,main,px -r %t4.o,g, -r %t4.o,g,px
; RUN: llvm-nm %t5.o.0 | FileCheck %s --check-prefix=NM0
; RUN: llvm-nm %t5.o.1 | FileCheck %s --check-prefix=NM1

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
define i32 @g() {
  ret i32 0
}
