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
; RUN:     -passes='function(coro-early),function(coro-elide),cgscc(coro-split),function(coro-cleanup)' 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-OPT

; note that we run CoroElidePass before CoroSplitPass. This is because CoroElidePass is part of
; function simplification pipeline, which runs before CoroSplitPass. And since @foo is not
; a coroutine, it won't be put back into the CGSCC, and hence won't trigger a CoroElidePass
; after CoroSplitPass.
; CHECK-ALL: CoroEarlyPass
; CHECK-OPT: CoroElidePass
; CHECK-ALL: CoroSplitPass
; CHECK-ALL: CoroCleanupPass

define void @foo() {
  ret void
}
