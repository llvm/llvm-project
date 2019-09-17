; Test to ensure that dead alias are dropped by converting to a declaration
; RUN: opt -module-summary %s -o %t1.bc
; RUN: llvm-lto2 run %t1.bc -r %t1.bc,barAlias,x \
; RUN:   -r %t1.bc,bar,x -r %t1.bc,zed,px \
; RUN:   -r %t1.bc,var,x -r %t1.bc,varAlias,x \
; RUN:   -o %t2.o -save-temps

; Check that bar and barAlias were dropped to declarations
; RUN: llvm-dis %t2.o.1.1.promote.bc -o - | FileCheck %s --check-prefix=DROP
; DROP-DAG: declare void @bar()
; DROP-DAG: declare void @barAlias()
; DROP-DAG: @var = external global i32
; DROP-DAG: @varAlias = external global i32

; Check that 'barAlias' and 'varAlias' were not inlined.
; RUN: llvm-objdump -d %t2.o.1 | FileCheck %s
; CHECK:      zed:
; CHECK-NEXT:  {{.*}}  pushq
; CHECK-NEXT:  {{.*}}  callq   0
; CHECK-NEXT:   movq  (%rip), %rax

; Check that 'barAlias' and 'varAlias' produced as undefined.
; RUN: llvm-readelf --symbols %t2.o.1 | FileCheck %s --check-prefix=SYMBOLS
; SYMBOLS: NOTYPE  GLOBAL DEFAULT  UND barAlias
; SYMBOLS: NOTYPE  GLOBAL DEFAULT  UND varAlias
; SYMBOLS: FUNC    GLOBAL DEFAULT    2 zed

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@barAlias = alias void(), void()* @bar
define void @bar() {
  ret void
}

@var = global i32 99
@varAlias = alias i32, i32* @var

define i32 @zed() {
  call void @barAlias()
  %1 = load i32, i32* @varAlias, align 4
  ret i32 %1
}
