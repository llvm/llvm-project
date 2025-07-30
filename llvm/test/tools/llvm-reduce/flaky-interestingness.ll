; Test that there is no assertion if the reproducer is flaky
; RUN: rm -f %t
; RUN: llvm-reduce -j=1 --delta-passes=instructions --test %python --test-arg %p/Inputs/flaky-test.py --test-arg %t %s -o /dev/null 2>&1 | FileCheck %s

; Check no error with -skip-verify-interesting-after-counting-chunks
; RUN: rm -f %t
; RUN: llvm-reduce -j=1 -skip-verify-interesting-after-counting-chunks --delta-passes=instructions --test %python --test-arg %p/Inputs/flaky-test.py --test-arg %t %s -o /dev/null 2>&1 | FileCheck --allow-empty -check-prefix=QUIET %s

; CHECK: warning: input module no longer interesting after counting chunks
; CHECK-NEXT: note: the interestingness test may be flaky, or there may be an llvm-reduce bug
; CHECK-NEXT: note: use -skip-verify-interesting-after-counting-chunks to suppress this warning

; QUIET-NOT: warning
; QUIET-NOT: note
; QUIET-NOT: error

declare void @foo(i32)

define void @func() {
  call void @foo(i32 0)
  call void @foo(i32 1)
  call void @foo(i32 2)
  call void @foo(i32 3)
  call void @foo(i32 4)
  call void @foo(i32 5)
  ret void
}
