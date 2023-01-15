; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-link -S %t/1.ll %t/1-aux.ll -o - | FileCheck %s

;--- 1.ll
$c = comdat any

@v = global i32 0, comdat ($c)

; CHECK: @v = global i32 0, comdat($c)
; CHECK: @v3 = external global i32
; CHECK: @v2 = external dllexport global i32

;--- 1-aux.ll
$c = comdat any

@v2 = weak dllexport global i32 0, comdat ($c)
define ptr @f2() {
  ret ptr @v2
}

@v3 = weak alias i32, ptr @v2
define ptr @f3() {
  ret ptr @v3
}

