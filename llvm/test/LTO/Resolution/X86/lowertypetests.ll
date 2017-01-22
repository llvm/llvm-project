; RUN: opt -thinlto-bc -o %t %s
; RUN: llvm-lto2 -r %t,f,plx -r %t,foo,lx -r %t,foo,plx -o %t1 %t
; RUN: llvm-nm %t1.0 | FileCheck --check-prefix=MERGED %s
; RUN: llvm-nm %t1.1 | FileCheck %s

; MERGED: R __typeid_foo_global_addr
; CHECK: U __typeid_foo_global_addr

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = global i32 0, !type !0

define i1 @f(i8* %ptr) {
  %p = call i1 @llvm.type.test(i8* %ptr, metadata !"foo")
  ret i1 %p
}

declare i1 @llvm.type.test(i8* %ptr, metadata %typeid) nounwind readnone

!0 = !{i32 0, !"foo"}
