; RUN: opt -S -function-attrs < %s | FileCheck %s
; RUN: opt -S -passes=function-attrs < %s | FileCheck %s

define void @f() {
; CHECK-LABEL:  define void @f() #0 {
 call void @g() [ "unknown"() ]
 ret void
}

define void @g() {
; CHECK-LABEL:  define void @g() #0 {
 call void @f()
 ret void
}


; CHECK: attributes #0 = { nofree nosync nounwind }
