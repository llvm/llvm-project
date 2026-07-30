; RUN: split-file %s %t
; RUN: opt -module-summary %t/a.s -o %t/a.bc
; RUN: opt -module-summary %t/b.s -o %t/b.bc
; RUN: not llvm-lto2 run %t/a.bc %t/b.bc -o %t/out -save-temps 2>&1 \
; RUN:   -r=%t/a.bc,callee,px \
; RUN:   -r=%t/b.bc,callee,x  \
; RUN:   -r=%t/b.bc,caller,px

; TODO: As part of LTO, we check that types match, but *we don't yet check that
; attributes match!!! What should happen if we remove "dontcall-error" from the
; definition or declaration of @callee?

; CHECK: call to callee marked "dontcall-error"

;--- a.s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @callee() "dontcall-error" noinline {
  ret i32 42
}

;--- b.s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @callee() "dontcall-error"

define i32 @caller() {
entry:
  %0 = call i32 @callee()
  ret i32 %0
}
