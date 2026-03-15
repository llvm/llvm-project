; REQUIRES: x86-registered-target
; RUN: opt  -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract  -b -n 0 -o - %t | llvm-dis  | FileCheck --check-prefix=M0 %s
; RUN: llvm-modextract  -b -n 1 -o - %t | llvm-dis  | FileCheck --check-prefix=M1 %s

target triple = "x86_64-unknown-linux-gnu"

define ptr @source() {
  ret ptr @g
}

; M0: @g.84f59439b469192440047efc8de357fb = external hidden constant [1 x ptr]{{$}}
; M1: @g.84f59439b469192440047efc8de357fb = hidden constant [1 x ptr] [ptr @ok.84f59439b469192440047efc8de357fb]
@g = internal constant [1 x ptr] [
  ptr @ok
], !type !0

; M0: define hidden i64 @ok.84f59439b469192440047efc8de357fb
; M1: define available_externally hidden i64 @ok.84f59439b469192440047efc8de357fb
define internal i64 @ok(ptr %this) {
  ret i64 42
}

!0 = !{i32 0, !"typeid"}
