; RUN: llvm-as < %s | llvm-dis - | FileCheck %s

; "uniform-work-group-size"="true" should be upgraded to a valueless attribute.
; CHECK: define void @true_val() #[[ATTR_TRUE:[0-9]+]]
define void @true_val() "uniform-work-group-size"="true" {
  ret void
}

; "uniform-work-group-size"="false" should be removed entirely.
; CHECK: define void @false_val() {
define void @false_val() "uniform-work-group-size"="false" {
  ret void
}

; Already-upgraded valueless attribute should be left alone.
; CHECK: define void @no_val() #[[ATTR_TRUE]]
define void @no_val() "uniform-work-group-size" {
  ret void
}

; CHECK-DAG: attributes #[[ATTR_TRUE]] = { "uniform-work-group-size" }
