; Tests whether the cache is sensitive to the dso-local bit on referenced
; globals.
; RUN: rm -rf %t.cache
; RUN: opt -module-hash -module-summary -o %t.bc %s
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache \
; RUN:   -r %t.bc,foo,px \
; RUN:   -r %t.bc,bar,px
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache \
; RUN:   -r %t.bc,foo,plx \
; RUN:   -r %t.bc,bar,px
; RUN: ls %t.cache | count 2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define weak void @foo() {
  ret void
}

define weak void()* @bar() {
  ret void()* @foo
}
