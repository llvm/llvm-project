; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo
;
; Test multiple global ifuncs to ensure iteration works correctly.

source_filename = "multiple_ifuncs.ll"

; Define resolver functions
define ptr @resolver1() {
entry:
  ret ptr @impl1
}

define ptr @resolver2() {
entry:
  ret ptr @impl2
}

define ptr @resolver3() {
entry:
  ret ptr @impl3
}

; Implementation functions
define i32 @impl1(i32 %x) {
  ret i32 %x
}

define i32 @impl2(i32 %x) {
  %1 = mul i32 %x, 2
  ret i32 %1
}

define i32 @impl3(i32 %x) {
  %1 = mul i32 %x, 3
  ret i32 %1
}

; Multiple ifuncs to test iteration
@ifunc1 = ifunc i32 (i32), ptr @resolver1
@ifunc2 = ifunc i32 (i32), ptr @resolver2
@ifunc3 = ifunc i32 (i32), ptr @resolver3
