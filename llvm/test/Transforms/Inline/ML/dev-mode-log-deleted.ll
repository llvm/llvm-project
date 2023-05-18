; REQUIRES: have_tflite
; RUN: opt -enable-ml-inliner=development -passes=scc-oz-module-inliner \
; RUN:     -training-log=%t -S < %s 
; RUN: %python %S/../../../../lib/Analysis/models/log_reader.py %t | FileCheck %s 

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

; CHECK: observation: 0
; CHECK: inlining_decision: 1
; CHECK: observation: 1
; CHECK: inlining_decision: 1
