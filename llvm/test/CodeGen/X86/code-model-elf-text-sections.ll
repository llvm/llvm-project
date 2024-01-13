; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=small -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SMALL
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=medium -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SMALL
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=large -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=LARGE

; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=small -function-sections -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SMALL-DS
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=medium -function-sections -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SMALL-DS
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=large -function-sections -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=LARGE-DS

; SMALL: .text {{.*}} AX {{.*}}
; SMALL-DS: .text.func {{.*}} AX {{.*}}
; LARGE: .ltext {{.*}} AXl {{.*}}
; LARGE-DS: .ltext.func {{.*}} AXl {{.*}}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

define void @func() {
  ret void
}
