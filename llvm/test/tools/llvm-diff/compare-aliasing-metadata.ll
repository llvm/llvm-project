; RUN: split-file %s %t
; RUN: llvm-diff %t/a.ll %t/a.ll
; RUN: not llvm-diff %t/a.ll %t/b.ll 2>&1 | FileCheck %s --implicit-check-not=same

; CHECK-LABEL: in function changed_tag:
; CHECK: > {{.*}}load i32
; CHECK-NEXT: !tbaa !{!{"float"
; CHECK: < {{.*}}load i32
; CHECK-NEXT: !tbaa !{!{"int"
; CHECK-LABEL: in function missing_tag:
; CHECK: > {{.*}}load i32
; CHECK-NOT: !tbaa
; CHECK: < {{.*}}load i32
; CHECK-NEXT: !tbaa !{!{"int"
; CHECK-LABEL: in function changed_offset:
; CHECK: > {{.*}}load i32
; CHECK-NEXT: !tbaa {{.*}}i64 4}{{$}}
; CHECK: < {{.*}}load i32
; CHECK-NEXT: !tbaa {{.*}}i64 0}{{$}}
; CHECK-LABEL: in function changed_scope:
; CHECK: > {{.*}}load i32
; CHECK-NEXT: !alias.scope {{.*}}"different"
; CHECK: < {{.*}}load i32
; CHECK-NEXT: !alias.scope {{.*}}"scope"
; CHECK-LABEL: in function changed_noalias:
; CHECK: > {{.*}}load i32
; CHECK-NEXT: !noalias {{.*}}"scope"
; CHECK: < {{.*}}load i32
; CHECK-NEXT: !noalias {{.*}}"other"
; CHECK-LABEL: in function same_tag_other_difference:
; CHECK-NOT: !tbaa !{
; CHECK-LABEL: in function different_opcode:
; CHECK-NOT: !tbaa !{
; CHECK-LABEL: in function same_tag_changed_noalias:
; CHECK-NOT: !tbaa !{
; CHECK: > {{.*}}load i32
; CHECK-NEXT: !noalias !{
; CHECK-NOT: !tbaa !{
; CHECK: < {{.*}}load i32
; CHECK-NEXT: !noalias !{
; CHECK-NOT: !tbaa !{

;--- a.ll
define i32 @same_tag(ptr %p) {
  %v = load i32, ptr %p, !tbaa !0
  ret i32 %v
}

define i32 @same_scope(ptr %p) {
  %v = load i32, ptr %p, !alias.scope !10, !noalias !20
  ret i32 %v
}

define i32 @changed_tag(ptr %p) {
  %v = load i32, ptr %p, !tbaa !0
  ret i32 %v
}

define i32 @missing_tag(ptr %p) {
  %v = load i32, ptr %p, !tbaa !0
  ret i32 %v
}

define i32 @changed_offset(ptr %p) {
  %v = load i32, ptr %p, !tbaa !4
  ret i32 %v
}

define i32 @changed_scope(ptr %p) {
  %v = load i32, ptr %p, !alias.scope !10
  ret i32 %v
}

define i32 @changed_noalias(ptr %p) {
  %v = load i32, ptr %p, !noalias !20
  ret i32 %v
}

define i32 @same_tag_other_difference(ptr %p, ptr %q) {
  %v = load i32, ptr %p, !tbaa !0
  %w = load i32, ptr %q, !tbaa !0
  %s = add i32 %v, %w
  ret i32 %s
}

define i32 @different_opcode(ptr %p) {
  %v = load i32, ptr %p, !tbaa !0
  ret i32 %v
}

define i32 @same_tag_changed_noalias(ptr %p) {
  %v = load i32, ptr %p, !tbaa !0, !noalias !20
  ret i32 %v
}

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !1, i64 0}
!5 = !{!"S", !1, i64 0, !1, i64 4}
!10 = !{!11}
!11 = distinct !{!11, !12, !"scope"}
!12 = distinct !{!12, !"domain"}
!20 = !{!21}
!21 = distinct !{!21, !12, !"other"}

;--- b.ll
define i32 @same_tag(ptr %p) {
  %v = load i32, ptr %p, !tbaa !0
  ret i32 %v
}

define i32 @same_scope(ptr %p) {
  %v = load i32, ptr %p, !alias.scope !10, !noalias !20
  ret i32 %v
}

define i32 @changed_tag(ptr %p) {
  %v = load i32, ptr %p, !tbaa !6
  ret i32 %v
}

define i32 @missing_tag(ptr %p) {
  %v = load i32, ptr %p
  ret i32 %v
}

define i32 @changed_offset(ptr %p) {
  %v = load i32, ptr %p, !tbaa !4
  ret i32 %v
}

define i32 @changed_scope(ptr %p) {
  %v = load i32, ptr %p, !alias.scope !30
  ret i32 %v
}

define i32 @changed_noalias(ptr %p) {
  %v = load i32, ptr %p, !noalias !10
  ret i32 %v
}

define i32 @same_tag_other_difference(ptr %p, ptr %q) {
  %v = load i32, ptr %p, !tbaa !0
  %w = load i32, ptr %p, !tbaa !0
  %s = add i32 %v, %w
  ret i32 %s
}

define i32 @different_opcode(ptr %p) {
  store i32 0, ptr %p, !tbaa !6
  ret i32 0
}

define i32 @same_tag_changed_noalias(ptr %p) {
  %v = load i32, ptr %p, !tbaa !0, !noalias !10
  ret i32 %v
}

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !1, i64 4}
!5 = !{!"S", !1, i64 0, !1, i64 4}
!6 = !{!7, !7, i64 0}
!7 = !{!"float", !2, i64 0}
!10 = !{!11}
!11 = distinct !{!11, !12, !"scope"}
!12 = distinct !{!12, !"domain"}
!20 = !{!21}
!21 = distinct !{!21, !12, !"other"}
!30 = !{!31}
!31 = distinct !{!31, !12, !"different"}
