; REQUIRES: asserts
; RUN: opt -passes=elim-avail-extern -avail-extern-to-local -stats -S 2>&1 < %s | FileCheck %s


declare void @call_out(ptr %fct)

define available_externally hidden void @f() {
  ret void
}

define available_externally hidden void @g() {
  ret void
}

define void @hello(ptr %g) {
  call void @f()
  %f = load ptr, ptr @f
  call void @call_out(ptr %f)
  ret void
}

; CHECK: define internal void @f.__uniq.{{[0-9|a-f]*}}()
; CHECK: declare hidden void @g()
; CHECK: call void @f.__uniq.{{[0-9|a-f]*}}()
; CHECK-NEXT: load ptr, ptr @f
; CHECK-NEXT: call void @call_out(ptr %f)
; CHECK: Statistics Collected
; CHECK: 1 elim-avail-extern - Number of functions converted
; CHECK: 1 elim-avail-extern - Number of functions removed