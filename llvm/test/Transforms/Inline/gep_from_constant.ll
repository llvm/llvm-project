; RUN: opt < %s -passes="print<inline-cost>" 2>&1 | FileCheck %s

; CHECK-LABEL: @foo
; CHECK: cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}, simplified to ptr inttoptr (i64 754974760 to ptr)

define ptr @foo(i64 %0) {
  %2 = inttoptr i64 754974720 to ptr
  %3 = getelementptr ptr addrspace(1), ptr %2, i64 %0
  ret ptr %3
}

define ptr @main() {
  %1 = call ptr @foo(i64 5)
  ret ptr %1
}
