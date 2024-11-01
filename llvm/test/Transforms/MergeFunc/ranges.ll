; RUN: opt -passes=mergefunc -S < %s | FileCheck %s
define i1 @cmp_with_range(i8*, i8*) {
  %v1 = load i8, i8* %0, !range !0
  %v2 = load i8, i8* %1, !range !0
  %out = icmp eq i8 %v1, %v2
  ret i1 %out
}

define i1 @cmp_no_range(i8*, i8*) {
; CHECK-LABEL: @cmp_no_range
; CHECK-NEXT: %v1 = load i8, i8* %0
; CHECK-NEXT:  %v2 = load i8, i8* %1
; CHECK-NEXT:  %out = icmp eq i8 %v1, %v2
; CHECK-NEXT:  ret i1 %out
  %v1 = load i8, i8* %0
  %v2 = load i8, i8* %1
  %out = icmp eq i8 %v1, %v2
  ret i1 %out
}

define i1 @cmp_different_range(i8*, i8*) {
; CHECK-LABEL: @cmp_different_range
; CHECK-NEXT:  %v1 = load i8, i8* %0, align 1, !range !1
; CHECK-NEXT:  %v2 = load i8, i8* %1, align 1, !range !1
; CHECK-NEXT:  %out = icmp eq i8 %v1, %v2
; CHECK-NEXT:  ret i1 %out
  %v1 = load i8, i8* %0, !range !1
  %v2 = load i8, i8* %1, !range !1
  %out = icmp eq i8 %v1, %v2
  ret i1 %out
}

define i1 @cmp_with_same_range(i8*, i8*) {
; CHECK-LABEL: @cmp_with_same_range
; CHECK: tail call i1 @cmp_with_range
  %v1 = load i8, i8* %0, !range !0
  %v2 = load i8, i8* %1, !range !0
  %out = icmp eq i8 %v1, %v2
  ret i1 %out
}

!0 = !{i8 0, i8 2}
!1 = !{i8 5, i8 7}
