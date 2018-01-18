; RUN: llvm-as -o %t.o %s
; RUN: llvm-lto2 run -o %t2.o %t.o -r=%t.o,_start,plx -r=%t.o,foobar,x
; RUN: llvm-readelf --symbols %t2.o.0 | FileCheck %s

; We used to fail the verifier by clearing dso_local from foobar

; CHECK:  0000000000000000     0 NOTYPE  GLOBAL HIDDEN   UND foobar

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foobar = external hidden global i32
define i32* @_start() {
  ret i32* @foobar
}
