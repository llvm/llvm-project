;; Check that parallel code generation keeps the code generation remarks of
;; every partition in the remark file, in partition order.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-lto2 run %t.bc -o %t.o -lto-partitions=2 \
; RUN:   -r %t.bc,foo,px -r %t.bc,bar,px \
; RUN:   -pass-remarks-output=%t.yaml -pass-remarks-filter=asm-printer
; RUN: llvm-nm %t.o.0 | FileCheck %s --check-prefix=PART0
; RUN: llvm-nm %t.o.1 | FileCheck %s --check-prefix=PART1
; RUN: FileCheck %s < %t.yaml

; RUN: llvm-lto2 run %t.bc -o %t.o -lto-partitions=2 \
; RUN:   -r %t.bc,foo,px -r %t.bc,bar,px \
; RUN:   -pass-remarks-output=%t.bitstream -pass-remarks-filter=asm-printer \
; RUN:   -pass-remarks-format=bitstream
; RUN: llvm-remarkutil bitstream2yaml %t.bitstream | FileCheck %s

;; The legacy LTOCodeGenerator, which takes its remark options separately.
; RUN: llvm-lto %t.bc -o %t.legacy.o -j2 \
; RUN:   -exported-symbol=foo -exported-symbol=bar \
; RUN:   -lto-pass-remarks-output=%t.legacy.yaml \
; RUN:   -lto-pass-remarks-filter=asm-printer
; RUN: FileCheck %s < %t.legacy.yaml

;; Partitions without functions have no remarks.
; RUN: llvm-lto2 run %t.bc -o %t.o -lto-partitions=4 \
; RUN:   -r %t.bc,foo,px -r %t.bc,bar,px \
; RUN:   -pass-remarks-output=%t.4.yaml -pass-remarks-filter=asm-printer
; RUN: FileCheck %s < %t.4.yaml

; PART0: T foo
; PART1: T bar

; CHECK:      Name: InstructionCount
; CHECK-NEXT: Function: foo
; CHECK:      Name: InstructionCount
; CHECK-NEXT: Function: bar

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() mustprogress {
  call void @bar()
  ret void
}

define void @bar() mustprogress {
  call void @foo()
  ret void
}
