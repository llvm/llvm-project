; REQUIRES: have_tf_api
; RUN: opt -enable-ml-inliner=development -passes=scc-oz-module-inliner \
; RUN:     -training-log=- -tfutils-text-log  -S < %s | FileCheck %s 

define i32 @top() {
    %a = call i32 @to_be_deleted()
    %b = call i32 @externally_visible()
    %ret = add i32 %a, %b
    ret i32 %ret
}

define internal i32 @to_be_deleted() {
    ret i32 1
}

define i32 @externally_visible() {
    ret i32 2
}

; CHECK:        key: "inlining_decision"
; CHECK-NEXT:   value {
; CHECK-NEXT:     feature {
; CHECK-NEXT:       int64_list {
; CHECK-NEXT:         value: 1
; CHECK-NEXT:       }
; CHECK-NEXT:     }
; CHECK-NEXT:     feature {
; CHECK-NEXT:       int64_list {
; CHECK-NEXT:         value: 1
; CHECK-NEXT:       }
; CHECK-NEXT:     }
; CHECK-NEXT:   }