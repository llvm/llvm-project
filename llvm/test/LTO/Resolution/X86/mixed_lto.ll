; Test mixed-mode LTO (mix of regular and thin LTO objects)
; RUN: opt %s -o %t1.o
; RUN: opt -module-summary %p/Inputs/mixed_lto.ll -o %t2.o

; RUN: llvm-lto2 -o %t3.o %t2.o %t1.o -r %t2.o,main,px -r %t2.o,g, -r %t1.o,g,px

; Task 0 is the regular LTO file (this file)
; RUN: llvm-nm %t3.o.0 | FileCheck %s --check-prefix=NM0
; NM0: T g

; Task 1 is the (first) ThinLTO file (Inputs/mixed_lto.ll)
; RUN: llvm-nm %t3.o.1 | FileCheck %s --check-prefix=NM1
; NM1-DAG: T main
; NM1-DAG: U g

target triple = "x86_64-unknown-linux-gnu"
define i32 @g() {
  ret i32 0
}
