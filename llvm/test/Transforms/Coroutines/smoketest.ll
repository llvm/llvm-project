; Test that all coroutine passes run in the correct order at all optimization
; levels adds coroutine passes to the pipeline.
;
; RUN: opt < %s -disable-output -passes='default<O0>' \
; RUN:     -debug-pass-manager 2>&1 | FileCheck %s --check-prefixes=CHECK-ALL
; RUN: opt < %s -disable-output -passes='default<O1>' \
; RUN:     -debug-pass-manager 2>&1 | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-OPT
; RUN: opt < %s -disable-output -passes='default<O2>' \
; RUN:     -debug-pass-manager 2>&1 | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-OPT
; RUN: opt < %s -disable-output -passes='default<O3>' \
; RUN:     -debug-pass-manager 2>&1 | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-OPT
; RUN: opt < %s -disable-output -debug-pass-manager \
; RUN:     -passes='module(coro-early),function(coro-elide),cgscc(coro-split),function(coro-cleanup)' 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-OPT

; note that we run CoroElidePass before CoroSplitPass. This is because CoroElidePass is part of
; function simplification pipeline, which runs before CoroSplitPass. And since @foo is not
; a coroutine, it won't be put back into the CGSCC, and hence won't trigger a CoroElidePass
; after CoroSplitPass.
; CHECK-ALL: CoroEarlyPass
; CHECK-OPT: CoroElidePass
; CHECK-ALL: CoroSplitPass
; CHECK-ALL: CoroCleanupPass

declare token @llvm.coro.id(i32, i8*, i8*, i8*)

define void @foo() {
  ret void
}
