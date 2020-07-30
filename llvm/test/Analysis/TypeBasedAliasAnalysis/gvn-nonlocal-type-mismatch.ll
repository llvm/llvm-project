; RUN: opt -tbaa -basic-aa -gvn -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

; GVN should ignore the store to p1 to see that the load from p is
; fully redundant.

; CHECK: @yes
; CHECK: if.then:
; CHECK-NEXT: store i32 0, i32* %q
; CHECK-NEXT: ret void

define void @yes(i1 %c, i32* %p, i32* %p1, i32* %q) nounwind {
entry:
  store i32 0, i32* %p, !tbaa !1
  store i32 1, i32* %p1, !tbaa !2
  br i1 %c, label %if.else, label %if.then

if.then:
  %t = load i32, i32* %p, !tbaa !1
  store i32 %t, i32* %q
  ret void

if.else:
  ret void
}

; GVN should ignore the store to p1 to see that the first load from p is
; fully redundant. However, the second load uses a different type. Theoretically
; the other type could be unified with the first type, however for now, GVN
; should just be conservative.

; CHECK: @watch_out_for_type_change
; CHECK: if.then:
; CHECK:   %t = load i32, i32* %p
; CHECK:   store i32 %t, i32* %q
; CHECK:   ret void
; CHECK: if.else:
; CHECK:   %u = load i32, i32* %p
; CHECK:   store i32 %u, i32* %q

define void @watch_out_for_type_change(i1 %c, i32* %p, i32* %p1, i32* %q) nounwind {
entry:
  store i32 0, i32* %p, !tbaa !1
  store i32 1, i32* %p1, !tbaa !2
  br i1 %c, label %if.else, label %if.then

if.then:
  %t = load i32, i32* %p, !tbaa !3
  store i32 %t, i32* %q
  ret void

if.else:
  %u = load i32, i32* %p, !tbaa !4
  store i32 %u, i32* %q
  ret void
}

; As before, but the types are swapped. This time GVN does managed to
; eliminate one of the loads before noticing the type mismatch.

; CHECK: @watch_out_for_another_type_change
; CHECK: if.then:
; CHECK:   store i32 0, i32* %q
; CHECK:   ret void
; CHECK: if.else:
; CHECK:   %u = load i32, i32* %p
; CHECK:   store i32 %u, i32* %q

define void @watch_out_for_another_type_change(i1 %c, i32* %p, i32* %p1, i32* %q) nounwind {
entry:
  store i32 0, i32* %p, !tbaa !1
  store i32 1, i32* %p1, !tbaa !2
  br i1 %c, label %if.else, label %if.then

if.then:
  %t = load i32, i32* %p, !tbaa !4
  store i32 %t, i32* %q
  ret void

if.else:
  %u = load i32, i32* %p, !tbaa !3
  store i32 %u, i32* %q
  ret void
}

!0 = !{}
!1 = !{!5, !5, i64 0}
!2 = !{!6, !6, i64 0}
!3 = !{!7, !7, i64 0}
!4 = !{!8, !8, i64 0}
!5 = !{!"red", !0}
!6 = !{!"blu", !0}
!7 = !{!"outer space", !9}
!8 = !{!"brick red", !5}
!9 = !{!"observable universe"}
