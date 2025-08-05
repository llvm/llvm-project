;; This test runs thin LTO with cache only to look for memory errors, either
;; as crashes or sanitizer errors. MCAsmStreamer has specific assumptions about
;; the lifetime of the output stream that are easy to overlook (see #138194).

; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -module-hash -module-summary -thinlto-bc %s -o %t1.bc
; RUN: llvm-lto2 run -cache-dir %t/cache --filetype=asm -o %t.o %t1.bc -r=%t1.bc,globalfunc

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @globalfunc() {
entry:
  ret void
}
