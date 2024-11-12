; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu \
; RUN:   -enable-machine-outliner -print-after=machine-outliner \
; RUN:   -o /dev/null \
; RUN:   2>&1 | FileCheck %s -check-prefixes=MIR

define i32 @empty_1() #0 {
  ret i32 1
}

; NOTE: only machine ir shall be printed
; MIR:	   RET64 killed $eax
; MIR-NOT: ret i32 1

