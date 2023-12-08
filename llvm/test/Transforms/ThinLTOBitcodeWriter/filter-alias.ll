; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -n 0 -o - %t | llvm-dis | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-modextract -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=CHECK1 %s
; CHECK0-NOT: @{{.*}}anon{{.*}}=
; CHECK0: @al = external global ptr
; CHECK0-NOT: @{{.*}}anon{{.*}}=
; CHECK1: @al = unnamed_addr alias ptr,

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

$al = comdat any

@anon = private unnamed_addr constant { [1 x ptr] } { [1 x ptr] [ptr null] }, comdat($al), !type !0

@al = external unnamed_addr alias ptr, getelementptr inbounds ({ [1 x ptr] }, ptr @anon, i32 0, i32 0, i32 1)

@foo = global i32 1

!0 = !{i64 8, !"?AVA@@"}
