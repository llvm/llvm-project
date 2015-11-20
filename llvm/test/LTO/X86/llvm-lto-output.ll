; Test the various output formats of the llvm-lto utility
;
; RUN: llvm-as < %s > %t1
;
; RUN: llvm-lto -exported-symbol=main -save-merged-module -filetype=asm -o %t2 %t1
; RUN: llvm-dis -o - %t2.merged.bc | FileCheck %s
; CHECK: @main()

; RUN: FileCheck --check-prefix=ASM %s < %t2
; RUN: llvm-lto -exported-symbol=main -filetype=obj -o %t2 %t1
; RUN: llvm-objdump -d %t2 | FileCheck --check-prefix=ASM %s
; ASM: main:
;

target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  ret i32 23
}

