; regresstion test for https://github.com/llvm/llvm-project/issues/60962
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 0 -o - %t | llvm-dis | FileCheck --check-prefix=M0 %s
; RUN: llvm-modextract -b -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=M1 %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@i = ifunc ptr (ptr, i64), ptr @hoge

@g = constant i8 1, !type !0
@assoc = private constant i8 2, !associated !1

define ptr @hoge() !type !2 {
bb:
  ret ptr null
}

; M0: @g = external constant
; M0: @i = ifunc ptr (ptr, i64), ptr @hoge
; M0: define ptr @hoge()
; M0-NOT: @assoc
; M1: @g = constant i8 1
; M1: @assoc = private constant i8 2
; M1-NOT: @i = ifunc ptr (ptr, i64), ptr @hoge
; M1-NOT: define ptr @hoge()

!0 = !{i32 0, !"typeid"}
!1 = !{ptr @g}
!2 = !{i64 0, !3}
!3 = distinct !{}
