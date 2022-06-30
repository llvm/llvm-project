; RUN: opt < %s -passes="print<inline-cost>" 2>&1 | FileCheck %s

; CHECK-LABEL: @foo
; CHECK: cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}, simplified to i8 addrspace(1)** inttoptr (i64 754974760 to i8 addrspace(1)**)

define i8 addrspace(1)** @foo(i64 %0) {
  %2 = inttoptr i64 754974720 to i8 addrspace(1)**
  %3 = getelementptr i8 addrspace(1)*, i8 addrspace(1)** %2, i64 %0
  ret i8 addrspace(1)** %3
}

define i8 addrspace(1)** @main() {
  %1 = call i8 addrspace(1)** @foo(i64 5)
  ret i8 addrspace(1)** %1
}
