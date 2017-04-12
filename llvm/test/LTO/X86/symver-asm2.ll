; Test to ensure symbol binding works correctly for symver directives,
; when the aliased symbols are defined in inline assembly, including
; cases when the symbol attributes are provided after the .symver
; directive.

; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -o %t2 %t1
; RUN: llvm-nm %t2 | FileCheck %s
; RUN: llvm-lto2 run -r %t1,_start,plx -r %t1,_start3,plx -r %t1,foo@@SOME_VERSION -r %t1,foo@SOME_VERSION3 -o %t3 %t1 -save-temps
; RUN: llvm-nm %t3.0 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".global _start"
module asm "_start:"
module asm "_start2:"
module asm "_start3:"
module asm ".symver _start, foo@@SOME_VERSION"
module asm ".symver _start2, foo@SOME_VERSION2"
module asm ".symver _start3, foo@SOME_VERSION3"
module asm ".local _start2"
module asm ".weak _start3"

; CHECK-DAG: T _start
; CHECK-DAG: t _start2
; CHECK-DAG: W _start3
; CHECK-DAG: T foo@@SOME_VERSION
; CHECK-DAG: t foo@SOME_VERSION2
; CHECK-DAG: W foo@SOME_VERSION3
