; RUN: split-file %s %t
; RUN: llvm-as < %t/valid.ll | llvm-dis | FileCheck %t/valid.ll
; RUN: not llvm-as < %t/exact-first.ll 2>&1 | FileCheck %t/exact-first.ll
; RUN: not llvm-as < %t/exact-twice.ll 2>&1 | FileCheck %t/exact-twice.ll
; RUN: not llvm-as < %t/exact-not-last.ll 2>&1 | FileCheck %t/exact-not-last.ll
; RUN: not llvm-as < %t/not-exact.ll 2>&1 | FileCheck %t/not-exact.ll

;--- valid.ll
; CHECK: declare ptr @lower_bound(i64) #[[LOWER:[0-9]+]]
declare ptr @lower_bound(i64) allocsize(0)
; CHECK: declare ptr @exact(i64) #[[EXACT:[0-9]+]]
declare ptr @exact(i64) allocsize(0, exact)
; CHECK: declare ptr @two_lower_bound(i64, i64) #[[TWOLOWER:[0-9]+]]
declare ptr @two_lower_bound(i64, i64) allocsize(0, 1)
; CHECK: declare ptr @two_exact(i64, i64) #[[TWOEXACT:[0-9]+]]
declare ptr @two_exact(i64, i64) allocsize(0, 1, exact)

; CHECK-LABEL: define void @call_sites(
; CHECK: call ptr @lower_bound(i64 8) #[[EXACT]]
; CHECK: call ptr @exact(i64 8) #[[LOWER]]
define void @call_sites() {
  call ptr @lower_bound(i64 8) allocsize(0, exact)
  call ptr @exact(i64 8) allocsize(0)
  ret void
}

; CHECK-DAG: attributes #[[LOWER]] = { allocsize(0) }
; CHECK-DAG: attributes #[[EXACT]] = { allocsize(0,exact) }
; CHECK-DAG: attributes #[[TWOLOWER]] = { allocsize(0,1) }
; CHECK-DAG: attributes #[[TWOEXACT]] = { allocsize(0,1,exact) }

;--- exact-first.ll
; CHECK: error: expected integer
declare ptr @f(i64) allocsize(exact, 0)

;--- exact-twice.ll
; CHECK: error: expected ')'
declare ptr @f(i64) allocsize(0, exact, exact)

;--- exact-not-last.ll
; CHECK: error: expected ')'
declare ptr @f(i64, i64) allocsize(0, exact, 1)

;--- not-exact.ll
; CHECK: error: expected 'exact'
declare ptr @f(i64, i64) allocsize(0, 1, nuw)
