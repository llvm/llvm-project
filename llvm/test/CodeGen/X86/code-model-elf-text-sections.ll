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
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=large -function-sections -basic-block-sections=all -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=LARGE-BB

; SMALL: .text {{.*}} AX {{.*}}
; SMALL: .ltext {{.*}} AXl {{.*}}
; SMALL: .ltext.2 {{.*}} AXl {{.*}}
; SMALL: .foo {{.*}} AX {{.*}}
; SMALL-DS: .text.func {{.*}} AX {{.*}}
; SMALL-DS: .ltext {{.*}} AXl {{.*}}
; SMALL-DS: .ltext.2 {{.*}} AXl {{.*}}
; SMALL-DS: .foo {{.*}} AX {{.*}}
; LARGE: .ltext {{.*}} AXl {{.*}}
; LARGE: .ltext.2 {{.*}} AXl {{.*}}
; LARGE: .foo {{.*}} AX {{.*}}
; LARGE-DS: .ltext.func {{.*}} AXl {{.*}}
; LARGE-DS: .ltext {{.*}} AXl {{.*}}
; LARGE-DS: .ltext.2 {{.*}} AXl {{.*}}
; LARGE-DS: .foo {{.*}} AX {{.*}}

; LARGE-BB: .ltext.func_bb {{.*}} AXl {{.*}}
; LARGE-BB: .ltext.func_bb {{.*}} AXl {{.*}}
; LARGE-BB: .ltext.func_bb {{.*}} AXl {{.*}}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

define void @func() {
  ret void
}

define void @ltext() section ".ltext" {
  ret void
}

define void @ltext2() section ".ltext.2" {
  ret void
}

define void @foo() section ".foo" {
  ret void
}

define i32 @func_bb(i32 %x) {
entry:
  %tobool = icmp ne i32 %x, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:
  ret i32 1

if.end:
  ret i32 0
}
