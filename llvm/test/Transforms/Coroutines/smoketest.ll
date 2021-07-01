; Test that all coroutine passes run in the correct order at all optimization
; levels and -enable-coroutines adds coroutine passes to the pipeline.
;
; RUN: opt < %s -disable-output -passes='default<O0>' -enable-coroutines \
; RUN:     -debug-pass-manager 2>&1 | FileCheck %s --check-prefixes=CHECK-ALL
; RUN: opt < %s -disable-output -passes='default<O1>' -enable-coroutines \
; RUN:     -debug-pass-manager 2>&1 | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-OPT
; RUN: opt < %s -disable-output -passes='default<O2>' -enable-coroutines \
; RUN:     -debug-pass-manager 2>&1 | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-OPT
; RUN: opt < %s -disable-output -passes='default<O3>' -enable-coroutines \
; RUN:     -debug-pass-manager 2>&1 | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-OPT
; RUN: opt < %s -disable-output -debug-pass-manager \
; RUN:     -passes='function(coro-early),cgscc(coro-split),function(coro-elide,coro-cleanup)' 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-OPT

; CHECK-ALL: CoroEarlyPass
; CHECK-ALL: CoroSplitPass
; CHECK-OPT: CoroElidePass
; CHECK-ALL: CoroCleanupPass

define void @foo() {
  ret void
}
