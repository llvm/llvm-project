; RUN: llvm-as %s -o %t1.o
; RUN: llvm-as %p/Inputs/alias-alias-1.ll -o %t2.o
; RUN: llvm-lto2 run -o %t3.o %t1.o %t2.o -r %t2.o,a, -r %t2.o,d,px -r %t1.o,a,p -r %t1.o,c,p -r %t1.o,b -save-temps
; RUN: llvm-dis < %t3.o.0.0.preopt.bc -o - | FileCheck %s
; RUN: FileCheck --check-prefix=RES %s < %t3.o.resolution.txt

; CHECK-NOT: alias
; CHECK: @c = global i32 1
; CHECK-NEXT: @d = global i32* @a
; CHECK-EMPTY:
; CHECK-NEXT: @a = weak alias i32, i32* @b
; CHECK-NEXT: @b = internal alias i32, i32* @c

; RES: 1.o{{$}}
; RES-NEXT: {{^}}-r={{.*}}1.o,c,p{{$}}
; RES-NEXT: {{^}}-r={{.*}}1.o,a,p{{$}}
; RES-NEXT: {{^}}-r={{.*}}1.o,b,{{$}}
; RES-NEXT: 2.o{{$}}
; RES-NEXT: {{^}}-r={{.*}}2.o,a,{{$}}
; RES-NEXT: {{^}}-r={{.*}}2.o,d,px{{$}}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = weak alias i32, i32* @b
@b = alias i32, i32* @c
@c = global i32 1
