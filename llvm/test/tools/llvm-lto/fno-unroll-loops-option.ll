; REQUIRES: asserts

; RUN: llvm-as < %s > %t1.bc

; Build with unrolling disabled (-lto-no-unroll-loops).
; RUN: llvm-lto %t1.bc -o %t.nounroll.o -lto-no-unroll-loops --exported-symbol=foo -save-merged-module
; RUN: llvm-dis  -o - %t.nounroll.o.merged.bc | FileCheck --check-prefix=NOUNROLL %s

; NOUNROLL: br label %loop
; NOUNROLL: br i1 %ec, label %exit, label %loop

; Build with unrolling enabled (by not passing -lto-no-unroll-loops). All
; branches should be gone.
; RUN: llvm-lto %t1.bc -o %t.nounroll.o --exported-symbol=foo -save-merged-module
; RUN: llvm-dis  -o - %t.nounroll.o.merged.bc | FileCheck --check-prefix=UNROLL %s

; UNROLL-NOT: br

define void @foo(i32* %ptr) {

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry], [ %iv.next, %loop ]
  %iv.ptr = getelementptr i32, i32* %ptr, i32 %iv
  store i32 %iv, i32* %iv.ptr
  %iv.next = add i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 10
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
