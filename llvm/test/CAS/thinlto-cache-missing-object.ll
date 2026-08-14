;; Check that a ThinLTO cache hit that points to an object which is missing from
;; the CAS is treated as a cache miss, instead of crashing.

;; The test CAS plugin is loaded by path, which assumes a 'lib' prefix.
; UNSUPPORTED: system-windows

; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -module-hash -module-summary %s -o %t/main.bc
; RUN: opt -module-hash -module-summary %p/Inputs/cache.ll -o %t/cache.bc

; DEFINE: %{plugin} = %llvmshlibdir/libCASPluginTest%pluginext

;; Populate the local CAS and "upload" it upstream, so that both the cache keys
;; and the objects they point to are available globally.
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t/cache.bc %t/main.bc \
; RUN:   -thinlto-cache-dir plugin:%{plugin}:%t/cas1?upstream-path=%t/upstream:no-logging=1

;; With a fresh local CAS the keys are still found upstream, but loading the
;; objects they point to reports them as missing. This has to be handled as a
;; cache miss and the modules recompiled.
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t/cache.bc %t/main.bc \
; RUN:   -thinlto-cache-dir plugin:%{plugin}:%t/cas2?upstream-path=%t/upstream:simulate-missing-objects=1:no-logging=1 \
; RUN:   -thinlto-save-objects %t/objects
; RUN: ls %t/objects | count 2

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @globalfunc() #0 {
entry:
  ret void
}
