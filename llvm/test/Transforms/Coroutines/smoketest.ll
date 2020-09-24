; Test that all coroutine passes run in the correct order at all optimization
; levels and -enable-coroutines adds coroutine passes to the pipeline.
;
; Legacy pass manager:
; RUN: opt < %s -disable-output -enable-coroutines -debug-pass=Arguments -O0 -enable-new-pm=0 2>&1 | FileCheck %s
; RUN: opt < %s -disable-output -enable-coroutines -debug-pass=Arguments -O1 -enable-new-pm=0 2>&1 | FileCheck %s
; RUN: opt < %s -disable-output -enable-coroutines -debug-pass=Arguments -O2 -enable-new-pm=0 2>&1 | FileCheck %s
; RUN: opt < %s -disable-output -enable-coroutines -debug-pass=Arguments -O3 -enable-new-pm=0 2>&1 | FileCheck %s
; RUN: opt < %s -disable-output -enable-coroutines -debug-pass=Arguments \
; RUN:     -coro-early -coro-split -coro-elide -coro-cleanup -enable-new-pm=0 2>&1 | FileCheck %s
; RUN: opt < %s -disable-output -debug-pass=Arguments -enable-new-pm=0 2>&1 \
; RUN:     | FileCheck %s -check-prefix=NOCORO
; New pass manager:
; RUN: opt < %s -disable-output -passes='default<O0>' -enable-coroutines \
; RUN:     -debug-pass-manager 2>&1 | FileCheck %s -check-prefix=NEWPM
; RUN: opt < %s -disable-output -passes='default<O1>' -enable-coroutines \
; RUN:     -debug-pass-manager 2>&1 | FileCheck %s -check-prefix=NEWPM
; RUN: opt < %s -disable-output -passes='default<O2>' -enable-coroutines \
; RUN:     -debug-pass-manager 2>&1 | FileCheck %s -check-prefix=NEWPM
; RUN: opt < %s -disable-output -passes='default<O3>' -enable-coroutines \
; RUN:     -debug-pass-manager 2>&1 | FileCheck %s -check-prefix=NEWPM
; RUN: opt < %s -disable-output -debug-pass-manager \
; RUN:     -passes='function(coro-early),cgscc(coro-split),function(coro-elide,coro-cleanup)' 2>&1 \
; RUN:     | FileCheck %s -check-prefix=NEWPM

; CHECK: coro-early
; CHECK: coro-split
; CHECK: coro-elide
; CHECK: coro-cleanup

; NOCORO-NOT: coro-early
; NOCORO-NOT: coro-split
; NOCORO-NOT: coro-elide
; NOCORO-NOT: coro-cleanup

; NEWPM: CoroEarlyPass
; NEWPM: CoroSplitPass
; NEWPM: CoroElidePass
; NEWPM: CoroCleanupPass

define void @foo() {
  ret void
}
