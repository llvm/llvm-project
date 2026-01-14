; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: 'alloc-variant-zeroed' must not be empty
declare ptr @a(i64) "alloc-variant-zeroed"=""

; CHECK: 'alloc-variant-zeroed' must not be empty
declare ptr @b(i64) "alloc-variant-zeroed"=""

; CHECK: 'alloc-variant-zeroed' must name a function belonging to the same 'alloc-family'
declare ptr @c(i64) "alloc-variant-zeroed"="c_zeroed" "alloc-family"="C"
declare ptr @c_zeroed(i64)

; CHECK: 'alloc-variant-zeroed' must name a function with 'allockind("zeroed")'
declare ptr @d(i64) "alloc-variant-zeroed"="d_zeroed" "alloc-family"="D"
declare ptr @d_zeroed(i64) "alloc-family"="D"

; CHECK: 'alloc-variant-zeroed' must name a function with the same signature
declare ptr @e(i64) "alloc-variant-zeroed"="e_zeroed" "alloc-family"="E"
declare ptr @e_zeroed(i64, i64) "alloc-family"="E" allockind("zeroed")

; CHECK: 'alloc-variant-zeroed' must name a function with the same calling convention
declare cc99 ptr @f(i64) "alloc-variant-zeroed"="f_zeroed" "alloc-family"="F"
declare cc100 ptr @f_zeroed(i64) "alloc-family"="F" allockind("zeroed")
