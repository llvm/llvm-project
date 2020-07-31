; RUN: llc -O0 < %s -mtriple armv7-linux-gnueabi -o - \
; RUN:   | llvm-mc -triple armv7-linux-gnueabi -filetype=obj -o - \
; RUN:    | llvm-readobj -r - | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7--linux-gnueabihf"

define internal i32 @arm_fn() #1 {
  %1 = tail call i32 @thumb_fn()
  ret i32 %1
}

define internal i32 @thumb_fn() #2 {
  %1 = tail call i32 @arm_fn()
  ret i32 %1
}

attributes #1 = { "target-features"="-thumb-mode" }
attributes #2 = { "target-features"="+thumb-mode" }

; CHECK: Relocations [
; CHECK-NEXT: Section (3) .rel.text {
; CHECK-NEXT: 0x0 R_ARM_JUMP24 thumb_fn 0x0
; CHECK-NEXT: 0x4 R_ARM_THM_JUMP24 arm_fn 0x0
; CHECK-NEXT: }
