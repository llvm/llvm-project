;; Test to ensure that we don't assert when checking variable importing
;; correctness for read only variables where the importing module contains
;; a linkonce_odr copy of the variable. In that case we do not need to
;; import the read only variable even when it has been exported from its
;; original module (in this case when we decided to import @bar).

; RUN: split-file %s %t
; RUN: opt -module-summary %t/foo.ll -o %t/foo.o
; RUN: opt -module-summary %t/bar.ll -o %t/bar.o
; RUN: llvm-lto2 run %t/foo.o %t/bar.o -r=%t/foo.o,foo,pl -r=%t/foo.o,bar,l -r=%t/foo.o,qux,pl -r=%t/bar.o,bar,pl -r=%t/bar.o,qux, -o %t.out -save-temps
; RUN: llvm-dis %t.out.1.3.import.bc -o - | FileCheck %s

;; Check that we have internalized @qux (since it is read only), and imported @bar.
; CHECK: @qux = internal global i32 0
; CHECK: define available_externally hidden void @bar()

;--- foo.ll

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@qux = linkonce_odr hidden global i32 0

define linkonce_odr hidden void @foo() {
  call void @bar()
  ret void
}

declare hidden void @bar()

;--- bar.ll

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@qux = linkonce_odr hidden global i32 0

define hidden void @bar() {
  %1 = load i32, i32* @qux, align 8
  ret void
}
