; Test that all coroutine passes run in the correct order at all optimization
; levels and -enable-coroutines adds coroutine passes to the pipeline.
;
; RUN: opt < %s -disable-output -enable-coroutines -debug-pass=Arguments -O0 2>&1 | FileCheck %s
; RUN: opt < %s -disable-output -enable-coroutines -debug-pass=Arguments -O1 2>&1 | FileCheck %s
; RUN: opt < %s -disable-output -enable-coroutines -debug-pass=Arguments -O2 2>&1 | FileCheck %s
; RUN: opt < %s -disable-output -enable-coroutines -debug-pass=Arguments -O3 2>&1 | FileCheck %s
; RUN: opt < %s -disable-output -enable-coroutines -debug-pass=Arguments \
; RUN:     -coro-early -coro-split -coro-elide -coro-cleanup 2>&1 | FileCheck %s
; RUN: opt < %s -disable-output -debug-pass=Arguments 2>&1 \
; RUN:     | FileCheck %s -check-prefix=NOCORO

; CHECK: coro-early
; CHECK: coro-split
; CHECK: coro-elide
; CHECK: coro-cleanup

; NOCORO-NOT: coro-early
; NOCORO-NOT: coro-split
; NOCORO-NOT: coro-elide
; NOCORO-NOT: coro-cleanup

define void @foo() {
  ret void
}
