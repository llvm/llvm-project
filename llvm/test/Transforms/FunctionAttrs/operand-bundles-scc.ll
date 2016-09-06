; RUN: opt -S -functionattrs < %s | FileCheck %s

define void @f() {
; CHECK-LABEL:  define void @f() {
 call void @g() [ "unknown"() ]
 ret void
}

define void @g() {
; CHECK-LABEL:  define void @g() {
 call void @f()
 ret void
}
