; RUN: opt -passes=early-cse -S < %s | FileCheck %s
; RUN: opt -passes=gvn -S < %s | FileCheck %s

define i1 @distinct_queries_are_not_commoned() {
; CHECK-LABEL: define i1 @distinct_queries_are_not_commoned() {
; CHECK-NEXT:    [[FIRST:%.*]] = call i1 @llvm.is.debugging.enabled()
; CHECK-NEXT:    [[SECOND:%.*]] = call i1 @llvm.is.debugging.enabled()
; CHECK-NEXT:    [[DIFFER:%.*]] = xor i1 [[FIRST]], [[SECOND]]
; CHECK-NEXT:    ret i1 [[DIFFER]]
;
  %first = call i1 @llvm.is.debugging.enabled()
  %second = call i1 @llvm.is.debugging.enabled()
  %differ = xor i1 %first, %second
  ret i1 %differ
}

declare i1 @llvm.is.debugging.enabled()
