; This test verifies the stable hash values for different global variables
; that have distinct names.
; We generate two different cgdata files from nearly identical outline instances,
; with the only difference being the last call target globals, @g vs @h.

; RUN: split-file %s %t

; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate=true -filetype=obj %t/local-g.ll -o %t/local-g.o
; RUN: llvm-cgdata --merge %t/local-g.o -o %t/local-g.cgdata
; RUN: llvm-cgdata --convert %t/local-g.cgdata -o %t/local-g.cgtext
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate=true -filetype=obj %t/local-h.ll -o %t/local-h.o
; RUN: llvm-cgdata --merge %t/local-h.o -o %t/local-h.cgdata
; RUN: llvm-cgdata --convert %t/local-h.cgdata -o %t/local-h.cgtext

; We compare the trees which are only different at the terminal node's hash value.
; Here we simply count the different lines that have `Hash` string.
; RUN: not diff %t/local-g.cgtext %t/local-h.cgtext 2>&1 | grep Hash | wc -l | FileCheck %s
; CHECK: 2

;--- local-g.ll
declare i32 @g(i32, i32, i32)
define i32 @f1() minsize {
  %1 = call i32 @g(i32 10, i32 1, i32 2);
  ret i32 %1
}
define i32 @f2() minsize {
  %1 = call i32 @g(i32 20, i32 1, i32 2);
  ret i32 %1
}

;--- local-h.ll
declare i32 @h(i32, i32, i32)
define i32 @f1() minsize {
  %1 = call i32 @h(i32 10, i32 1, i32 2);
  ret i32 %1
}
define i32 @f2() minsize {
  %1 = call i32 @h(i32 20, i32 1, i32 2);
  ret i32 %1
}
