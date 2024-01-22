; RUN: llc --relocation-model=pic < %s | FileCheck %s
; RUN: llc --relocation-model=static < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Jump tables shouldn't go through the GOT.
define i32 @jump_table(i32 %x) #0 {
; CHECK-LABEL: jump_table:
; CHECK-NOT: @GOT
  switch i32 %x, label %default [
    i32 0, label %1
    i32 1, label %2
    i32 2, label %3
    i32 3, label %4
  ]
1:
  ret i32 7
2:
  ret i32 42
3:
  ret i32 3
4:
  ret i32 8
default:
  ret i32 %x
}

attributes #0 = { "target-features"="+tagged-globals" }
