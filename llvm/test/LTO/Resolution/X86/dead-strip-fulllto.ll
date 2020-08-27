; RUN: opt -module-summary -o %t %s
; RUN: opt -module-summary -o %t2 %S/Inputs/dead-strip-fulllto.ll

; Adding '--pass-remarks-with-hotness' should not cause crash.
; RUN: llvm-lto2 run --pass-remarks-output=%t4.yaml --pass-remarks-filter=. --pass-remarks-with-hotness \
; RUN:               %t -r %t,main,px -r %t,live1, -r %t,live2,p -r %t,dead2,p \
; RUN:               %t2 -r %t2,live1,p -r %t2,live2, -r %t2,dead1,p -r %t2,dead2, -r %t2,odr, \
; RUN:               -save-temps -o %t3

; RUN: cat %t4.yaml | FileCheck %s -check-prefix=REMARK
; RUN: llvm-nm %t3.0 | FileCheck --check-prefix=FULL %s
; RUN: llvm-nm %t3.1 | FileCheck --check-prefix=THIN %s

; RUN: llvm-lto2 run %t -r %t,main,px -r %t,live1, -r %t,live2,p -r %t,dead2,p \
; RUN:               %t2 -r %t2,live1,p -r %t2,live2, -r %t2,dead1,p -r %t2,dead2, -r %t2,odr, \
; RUN:               -save-temps -o %t3 -O0
; RUN: llvm-nm %t3.0 | FileCheck --check-prefix=FULL %s
; RUN: llvm-nm %t3.1 | FileCheck --check-prefix=THIN %s

; REMARK:      Pass:            lto
; REMARK-NEXT: Name:            deadfunction
; REMARK-NEXT: Function:        dead2
; REMARK-NEXT: Args:
; REMARK-NEXT:   - Function:        dead2
; REMARK-NEXT:   - String:          ' not added to the combined module '

; FULL-NOT: dead
; FULL: U live1
; FULL: T live2
; FULL: T main

; THIN-NOT: dead
; THIN: T live1
; THIN: U live2
; THIN-NOT: odr

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @main() {
  call void @live1()
  ret void
}

declare void @live1()

define void @live2() {
  ret void
}

define void @dead2() {
  ret void
}

!0 = !{i32 1, !"ThinLTO", i32 0}
!llvm.module.flags = !{ !0 }
