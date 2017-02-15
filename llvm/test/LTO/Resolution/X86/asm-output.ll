; Test the ability to emit assembly code from the resolution-based LTO API
;
; RUN: llvm-as < %s > %t1.bc
;
; RUN: llvm-lto2 -filetype=asm -r %t1.bc,main,px -o %t2 %t1.bc
; RUN: FileCheck --check-prefix=ASM %s < %t2.0
; RUN: llvm-lto2 -filetype=obj -r %t1.bc,main,px -o %t2 %t1.bc
; RUN: llvm-objdump -d %t2.0 | FileCheck --check-prefix=ASM %s
;
; ASM: main:

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  ret i32 23
}

