; RUN: llvm-as %s -o %t1.o
; RUN: llvm-as %p/Inputs/intrinsic.ll -o %t2.o
; RUN: llvm-lto2 run -o %t3.o %t1.o %t2.o -r %t1.o,foo

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
%foo = type {  }
declare void @foo( ptr  )
