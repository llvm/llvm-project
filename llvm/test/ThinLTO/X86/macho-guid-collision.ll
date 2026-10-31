; RUN: split-file %s %t
; RUN: opt -module-summary %t/defs.ll -o %t/defs.bc
; RUN: opt -module-summary %t/main.ll -o %t/main.bc
; RUN: llvm-lto2 run %t/defs.bc %t/main.bc -o %t/out -save-temps \
; RUN:   -import-instr-limit=0 \
; RUN:   -r=%t/defs.bc,_foo,pl -r=%t/defs.bc,__foo,pl \
; RUN:   -r=%t/main.bc,_main,plx -r=%t/main.bc,_foo,l \
; RUN:   -r=%t/main.bc,__foo,l
; RUN: llvm-dis %t/out.1.3.import.bc -o - | FileCheck %s

;; Global resolutions use mangled symbol names, whereas the summary records IR
;; names. Looking up the IR name _foo in the resolutions for Mach-O incorrectly
;; assigns its GUID to foo and allows foo to be internalized.
; CHECK: define dso_local i32 @foo()
; CHECK: define dso_local i32 @_foo()

;--- defs.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

define i32 @foo() noinline {
  ret i32 1
}

define i32 @_foo() noinline {
  ret i32 2
}

;--- main.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

declare i32 @foo()
declare i32 @_foo()

define i32 @main() {
  %a = call i32 @foo()
  %b = call i32 @_foo()
  %sum = add i32 %a, %b
  ret i32 %sum
}
