; RUN: opt < %s -mtriple=x86_64-unknown-unknown -passes=mergeicmps -verify-dom-info -S | FileCheck %s

; Merges adjacent comparisons with constants even if only in single basic block

define i1 @merge_single(ptr nocapture noundef readonly dereferenceable(2) %p) {
; CHECK-LABEL: @merge_single(
; CHECK:       entry:
; CHECK-NEXT:   [[TMP0:%.*]] = getelementptr inbounds i8, ptr [[P:%.*]], i64 1
; CHECK-NEXT:   [[TMP1:%.*]] = alloca [2 x i8], align 1
; CHECK-NEXT:   store [2 x i8] c"\FF\FF", ptr [[TMP1]], align 1
; CHECK-NEXT:   [[MEMCMP:%.*]] = call i32 @memcmp(ptr [[P]], ptr [[TMP1]], i64 2)
; CHECK-NEXT:   [[CMP0:%.*]] = icmp eq i32 [[MEMCMP]], 0
; CHECK-NEXT:   ret i1 [[CMP0]]
;
entry:
  %0 = load i8, ptr %p, align 1
  %arrayidx1 = getelementptr inbounds i8, ptr %p, i64 1
  %1 = load i8, ptr %arrayidx1, align 1
  %cmp = icmp eq i8 %0, -1
  %cmp3 = icmp eq i8 %1, -1
  %2 = select i1 %cmp, i1 %cmp3, i1 false
  ret i1 %2
}
