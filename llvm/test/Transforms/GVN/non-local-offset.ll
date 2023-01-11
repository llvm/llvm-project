; RUN: opt -passes=gvn -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

; GVN should ignore the store to p[1] to see that the load from p[0] is
; fully redundant.

; CHECK-LABEL: @yes(
; CHECK: if.then:
; CHECK-NEXT: store i32 0, ptr %q
; CHECK-NEXT: ret void

define void @yes(i1 %c, ptr %p, ptr %q) nounwind {
entry:
  store i32 0, ptr %p
  %p1 = getelementptr inbounds i32, ptr %p, i64 1
  store i32 1, ptr %p1
  br i1 %c, label %if.else, label %if.then

if.then:
  %t = load i32, ptr %p
  store i32 %t, ptr %q
  ret void

if.else:
  ret void
}

; GVN should ignore the store to p[1] to see that the first load from p[0] is
; fully redundant. However, the second load is larger, so it's not a simple
; redundancy.

; CHECK-LABEL: @watch_out_for_size_change(
; CHECK: if.then:
; CHECK-NEXT: store i32 0, ptr %q
; CHECK-NEXT: ret void
; CHECK: if.else:
; CHECK: load i64, ptr %p
; CHECK: store i64

define void @watch_out_for_size_change(i1 %c, ptr %p, ptr %q) nounwind {
entry:
  store i32 0, ptr %p
  %p1 = getelementptr inbounds i32, ptr %p, i64 1
  store i32 1, ptr %p1
  br i1 %c, label %if.else, label %if.then

if.then:
  %t = load i32, ptr %p
  store i32 %t, ptr %q
  ret void

if.else:
  %t64 = load i64, ptr %p
  store i64 %t64, ptr %q
  ret void
}
