; RUN: llvm-as %s -o %t1.o
; RUN: llvm-as %p/Inputs/start-lib1.ll -o %t2.o
; RUN: llvm-as %p/Inputs/start-lib2.ll -o %t3.o

; RUN: llvm-lto2 run -r %t1.o,_start,px -r %t2.o,foo,px -r %t3.o,bar,px -r %t2.o,bar,lx -o %t3 %t1.o %t2.o %t3.o
; RUN: llvm-readelf -S %t3.0 | FileCheck %s --implicit-check-not=.llvmbc

; RUN: llvm-lto2 run -r %t1.o,_start,px -r %t2.o,foo,px -r %t3.o,bar,px -r %t2.o,bar,lx -lto-embed-bitcode=none -o %t3 %t1.o %t2.o %t3.o
; RUN: llvm-readelf -S %t3.0 | FileCheck %s --implicit-check-not=.llvmbc

; RUN: llvm-lto2 run -r %t1.o,_start,px -r %t2.o,foo,px -r %t3.o,bar,px -r %t2.o,bar,lx -lto-embed-bitcode=optimized -o %t3 %t1.o %t2.o %t3.o
; RUN: llvm-readelf -S %t3.0 | FileCheck %s --check-prefix=CHECK-ELF
; RUN: llvm-objcopy --dump-section=.llvmbc=%t-embedded.bc %t3.0 /dev/null
; RUN: llvm-dis %t-embedded.bc -o - | FileCheck %s --check-prefixes=CHECK-LL,CHECK-OPT

; RUN: llvm-lto2 run -r %t1.o,_start,px -r %t2.o,foo,px -r %t3.o,bar,px -r %t2.o,bar,lx -lto-embed-bitcode=post-merge-pre-opt -o %t3 %t1.o %t2.o %t3.o
; RUN: llvm-readelf -S %t3.0 | FileCheck %s --check-prefix=CHECK-ELF
; RUN: llvm-objcopy --dump-section=.llvmbc=%t-embedded.bc %t3.0 /dev/null
; RUN: llvm-dis %t-embedded.bc -o - | FileCheck %s --check-prefixes=CHECK-LL,CHECK-NOOPT

; CHECK-ELF:      .text   PROGBITS 0000000000000000 [[#%x,OFF:]] [[#%x,SIZE:]] 00 AX 0
; CHECK-ELF-NEXT: .llvmbc PROGBITS 0000000000000000 [[#%x,OFF:]] [[#%x,SIZE:]] 00    0

; CHECK-LL: @_start
; CHECK-LL: @foo
; CHECK-OPT-NEXT: ret void
; CHECK-NOOPT-NEXT: call void @bar
; CHECK-LL: @bar

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_start() {
  ret void
}
