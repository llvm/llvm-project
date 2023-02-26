; REQUIRES: remote-cache-service

; RUN: opt -module-hash -module-summary %s -o %t.bc
; RUN: opt -module-hash -module-summary %p/Inputs/cache.ll -o %t2.bc

; RUN: rm -f %{remote-cache-dir}/%basename_t
; RUN: rm -rf %t && mkdir -p %t

; RUN: llvm-remote-cache-test -socket-path=%{remote-cache-dir}/%basename_t -cache-path=%t/cache -- \
; RUN:    llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir grpc:%{remote-cache-dir}/%basename_t

; RUN: llvm-cas --cas %t/cache/cas --dump | FileCheck %s

;; Check the cas dump, expect 4 items.
; CHECK: index:
; CHECK: root
; CHECK: records
; CHECK-NEXT: SK=datapool
; CHECK-NEXT: SK=datapool
; CHECK-NEXT: SK=datapool
; CHECK-NEXT: SK=datapool
; CHECK-EMPTY:
; CHECK-NEXT: pool:

; RUN: llvm-remote-cache-test -socket-path=%{remote-cache-dir}/%basename_t -cache-path=%t/cache -- \
; RUN:    llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir grpc:%{remote-cache-dir}/%basename_t

;; CAS should remain the same.
; RUN: llvm-cas --cas %t/cache/cas --dump | FileCheck %s

;; Check save object file path also works
; RUN: llvm-remote-cache-test -socket-path=%{remote-cache-dir}/%basename_t -cache-path=%t/cache -- \
; RUN:    llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir grpc:%{remote-cache-dir}/%basename_t -thinlto-save-objects %t/objects
; RUN: llvm-cas --cas %t/cache/cas --dump | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @globalfunc() #0 {
entry:
  ret void
}
