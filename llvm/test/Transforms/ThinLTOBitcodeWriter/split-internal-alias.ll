; REQUIRES: aarch64-registered-target
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 0 -o %t0 %t
; RUN: llvm-modextract -b -n 1 -o %t1 %t
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=M0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=M1 %s
; RUN: llvm-lto2 run %t -o %t.out \
; RUN:   -r=%t,use_vtable,plx \
; RUN:   -r=%t,vtable.69cc645c8be4ef4172cfb55c93f80acd,l \
; RUN:   -r=%t,rtti_alias.69cc645c8be4ef4172cfb55c93f80acd,plx \
; RUN:   -r=%t,rtti_alias.69cc645c8be4ef4172cfb55c93f80acd,l \
; RUN:   -r=%t,vtable.69cc645c8be4ef4172cfb55c93f80acd,plx

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; M0: @rtti_storage = private constant i8 1
@rtti_storage = private constant i8 1

; M0: @rtti_alias = internal alias i8, ptr @rtti_storage
; M0: @rtti_alias.69cc645c8be4ef4172cfb55c93f80acd = hidden alias i8, ptr @rtti_storage
; M0-NOT: hidden alias i8, ptr @rtti_alias{{$}}
@rtti_alias = internal alias i8, ptr @rtti_storage

; M1: @vtable = internal constant [1 x ptr] [ptr @rtti_alias.69cc645c8be4ef4172cfb55c93f80acd], !type !0
@vtable = internal constant [1 x ptr] [ptr @rtti_alias], !type !0

define ptr @use_vtable() {
  ret ptr @vtable
}

!0 = !{i64 0, !"_ZTS4Base"}
