; RUN: opt -passes='default<O1>' -S < %s | FileCheck %s

; The three query sites must not be merged by CFG simplification.
define void @preserve_query_sites(i32 %selector) {
; CHECK-LABEL: define void @preserve_query_sites(
; CHECK:       first:
; CHECK-NEXT:    call i1 @llvm.is.debugging.enabled()
; CHECK:       second:
; CHECK-NEXT:    call i1 @llvm.is.debugging.enabled()
; CHECK:       join:
; CHECK-NEXT:    call i1 @llvm.is.debugging.enabled()
;
entry:
  switch i32 %selector, label %join [
    i32 5, label %first
    i32 7, label %second
  ]

first:
  %first.query = call i1 @llvm.is.debugging.enabled()
  br label %join

second:
  %second.query = call i1 @llvm.is.debugging.enabled()
  br label %join

join:
  %join.query = call i1 @llvm.is.debugging.enabled()
  ret void
}

; The equivalent unannotated calls establish that the pipeline exercises the
; merge opportunity used above.
define void @mergeable_control(i32 %selector) {
; CHECK-LABEL: define void @mergeable_control(
; CHECK-COUNT-1: call i1 @ordinary.query()
;
entry:
  switch i32 %selector, label %join [
    i32 5, label %first
    i32 7, label %second
  ]

first:
  %first.query = call i1 @ordinary.query()
  br label %join

second:
  %second.query = call i1 @ordinary.query()
  br label %join

join:
  %join.query = call i1 @ordinary.query()
  ret void
}

declare i1 @llvm.is.debugging.enabled()
declare i1 @ordinary.query()
