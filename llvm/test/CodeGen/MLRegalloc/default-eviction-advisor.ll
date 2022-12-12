; Check that, in the absence of dependencies, we emit an error message when
; trying to use ML-driven advisor.
; REQUIRES: !have_tf_aot
; REQUIRES: !have_tf_api
; REQUIRES: default_triple
; RUN: not llc -O2 -regalloc-enable-advisor=development < %s 2>&1 | FileCheck %s
; RUN: not llc -O2 -regalloc-enable-advisor=release < %s 2>&1 | FileCheck %s
; RUN: llc -O2 -regalloc-enable-advisor=default < %s 2>&1 | FileCheck %s --check-prefix=DEFAULT

; regalloc-enable-advisor is not enabled for NVPTX
; UNSUPPORTED: target=nvptx{{.*}}

define void @f2(i64 %lhs, i64 %rhs, i64* %addr) {
  %sum = add i64 %lhs, %rhs
  store i64 %sum, i64* %addr
  ret void
}

; CHECK: Requested regalloc eviction advisor analysis could be created. Using default
; DEFAULT-NOT: Requested regalloc eviction advisor analysis could be created. Using default
