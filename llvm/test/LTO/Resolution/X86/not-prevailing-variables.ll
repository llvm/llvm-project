; RUN: opt -module-summary %s -o %t1.o
; RUN: llvm-lto2 run -save-temps -o %t2.o %t1.o      \
; RUN:   -r %t1.o,testVar1,plx -r %t1.o,testVar2,plx \
; RUN:   -r %t1.o,var1,pl -r %t1.o,var2,lx

; Test contains two retainedNodes: var1 and var2.
; var2 is not prevailing and here we check it is not inlined.

; Check 'var2' was not inlined.
; RUN: llvm-objdump -d %t2.o.1 | FileCheck %s
; CHECK:      <testVar1>:
; CHECK-NEXT:   movl $10, %eax
; CHECK-NEXT:   retq
; CHECK:      <testVar2>:
; CHECK-NEXT:   movl  (%rip), %eax
; CHECK-NEXT:   retq

; Check 'var2' is undefined.
; RUN: llvm-readelf --symbols %t2.o.1 | FileCheck %s --check-prefix=UND
; UND: NOTYPE  GLOBAL DEFAULT UND var2

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@var1 = global i32 10, align 4
define i32 @testVar1() {
  %1 = load i32, i32* @var1, align 4
  ret i32 %1
}

@var2 = global i32 11, align 4
define i32 @testVar2() {
  %1 = load i32, i32* @var2, align 4
  ret i32 %1
}
