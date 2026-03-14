; RUN: opt -module-summary %s -o %t1.o
; RUN: opt -module-summary -o %t2.o %S/Inputs/not-prevailing.ll
; RUN: llvm-lto2 run -o %t3.o %t1.o %t2.o -r %t1.o,foo,x -r %t1.o,zed,px -r %t1.o,bar,x \
; RUN:   -r %t2.o,bar,x -save-temps

; Check that 'foo' and 'bar' were not inlined.
; CHECK:      <zed>:
; CHECK-NEXT:  {{.*}}  pushq   %rbx
; CHECK-NEXT:  {{.*}}  callq   0x6 <zed+0x6>
; CHECK-NEXT:  {{.*}}  movl    %eax, %ebx
; CHECK-NEXT:  {{.*}}  callq   0xd <zed+0xd>
; CHECK-NEXT:  {{.*}}  movl    %ebx, %eax
; CHECK-NEXT:  {{.*}}  popq    %rbx
; CHECK-NEXT:  {{.*}}  retq

; RUN: llvm-objdump -d %t3.o.1 | FileCheck %s
; RUN: llvm-readelf --symbols %t3.o.1 | FileCheck %s --check-prefix=SYMBOLS

; Check that 'foo' and 'bar' produced as undefined.
; SYMBOLS: FUNC    GLOBAL DEFAULT    2 zed
; SYMBOLS: NOTYPE  GLOBAL DEFAULT  UND foo
; SYMBOLS: NOTYPE  GLOBAL DEFAULT  UND bar

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$foo = comdat any
define weak i32 @foo() comdat {
  ret i32 65
}

declare void @bar()

define i32 @zed() {
  %1 = tail call i32 @foo()
  call void @bar()
  ret i32 %1
}
