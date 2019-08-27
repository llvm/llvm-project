; Check we don't crash when linking a global variable with appending linkage
; if the types in their elements don't have a straightforward mapping, forcing
; us to use bitcasts.

; RUN: opt %s -o %t1.o
; RUN: opt %p/Inputs/appending-var-2.ll -o %t2.o

; RUN: llvm-lto2 run -o %t3.o %t1.o %t2.o -r %t1.o,bar, -r %t2.o,bar,px

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"foo.1" = type { i8, i8 }
declare dso_local i32 @bar(%"foo.1"* nocapture readnone %this) local_unnamed_addr

@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (%"foo.1"*)* @bar to i8*)], section "llvm.metadata"
