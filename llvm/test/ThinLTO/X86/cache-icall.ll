; Test that the list of CFI jumptable entries is part of ThinLTO cache key.

; Linking Inputs/cache-icall.ll results in f() being added to CFI jumptable; otherwise it is not.
; This affects code generated for any users of f(). Make sure that we don't pull a stale object
; file for %t.o from the cache.

; RUN: opt -module-hash -module-summary -thinlto-bc -thinlto-split-lto-unit %s -o %t.bc
; RUN: opt -module-hash -module-summary -thinlto-bc -thinlto-split-lto-unit %p/Inputs/cache-icall.ll -o %t2.bc

; RUN: rm -Rf %t.cache && mkdir %t.cache

; RUN: llvm-lto2 run -o %t-no.o %t.bc -cache-dir %t.cache \
; RUN:   -r=%t.bc,_start,px \
; RUN:   -r=%t.bc,f,

; RUN: llvm-readelf -symbols %t-no.o.* | FileCheck %s --check-prefix=SYMBOLS-NO

; RUN: llvm-lto2 run -o %t-yes.o %t.bc %t2.bc -cache-dir %t.cache \
; RUN:   -r=%t.bc,_start,px \
; RUN:   -r=%t.bc,f, \
; RUN:   -r=%t2.bc,f,p

; RUN: llvm-readelf -symbols %t-yes.o.* | FileCheck %s --check-prefix=SYMBOLS-YES

; SYMBOLS-NO-DAG: {{FUNC .* f.cfi_jt$}}
; SYMBOLS-NO-DAG: {{NOTYPE .* UND f.cfi_jt$}}

; SYMBOLS-YES-NOT: f.cfi_jt
; SYMBOLS-YES-DAG: {{FUNC .* f.cfi$}}
; SYMBOLS-YES-DAG: {{NOTYPE .* UND f.cfi$}}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i8* @_start(void ()* %p) !type !0 {
entry:
  %0 = bitcast void ()* %p to i8*
  %1 = tail call i1 @llvm.type.test(i8* %0, metadata !"_ZTSFvvE")
  br i1 %1, label %cont, label %trap

trap:                                             ; preds = %entry
  tail call void @llvm.trap()
  unreachable

cont:                                             ; preds = %entry
  tail call void %p()
  ret i8* bitcast (void ()* @f to i8*)
}

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.trap()
declare !type !1 void @f()

!0 = !{i64 0, !"_ZTSFPvPFvvEE"}
!1 = !{i64 0, !"_ZTSFvvE"}
