; RUN: llvm-as %s -o %t1.o
; RUN: llvm-as %p/Inputs/alias-1.ll -o %t2.o
; RUN: llvm-lto2 run -o %t3.o %t2.o %t1.o -r %t2.o,a,px -r %t1.o,a, -r %t1.o,b,px -save-temps
; RUN: llvm-dis < %t3.o.0.0.preopt.bc -o - | FileCheck %s
; RUN: FileCheck --check-prefix=RES %s < %t3.o.resolution.txt

; CHECK-NOT: alias
; CHECK: @a = global i32 42
; CHECK-NEXT: @b = global i32 1
; CHECK-NOT: alias

; RES: 2.o{{$}}
; RES: {{^}}-r={{.*}}2.o,a,px{{$}}
; RES: 1.o{{$}}
; RES: {{^}}-r={{.*}}1.o,b,px{{$}}
; RES: {{^}}-r={{.*}}1.o,a,{{$}}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = weak alias i32, i32* @b
@b = global i32 1
