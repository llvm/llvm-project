; RUN: llc %s -o %t.o -filetype=obj -relocation-model=pic
; RUN: llvm-readobj -r %t.o | FileCheck %s
; CHECK:      Section ({{.}}) .rela.text {
; CHECK-NEXT:   0x3 R_X86_64_REX_GOTPCRELX a 0xFFFFFFFFFFFFFFFC
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external global i32

define i32 @f() {
  %t = load i32, i32* @a
  ret i32 %t
}
