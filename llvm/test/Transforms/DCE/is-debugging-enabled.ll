; RUN: opt -passes=dce -S < %s | FileCheck %s

define void @unused_query_is_observable() {
; CHECK-LABEL: define void @unused_query_is_observable() {
; CHECK-NEXT:    [[ENABLED:%.*]] = call i1 @llvm.is.debugging.enabled()
; CHECK-NEXT:    ret void
;
  %enabled = call i1 @llvm.is.debugging.enabled()
  ret void
}

declare i1 @llvm.is.debugging.enabled()
