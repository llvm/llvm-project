; REQUIRES: asserts
; RUN: opt -passes=slp-vectorizer -mtriple=aarch64 -slp-threshold=-100 -slp-use-vplan-codegen -debug-only=SLP -disable-output %s 2>&1 | FileCheck %s

define void @print_vplan(ptr %a, ptr %b, ptr %c) {
; CHECK-LABEL: SLP: VPlan for tree:
; CHECK-NEXT:  VPlan 'SLP tree for UF>=1' {
; CHECK-EMPTY:
; CHECK-NEXT:  {{.+}}:
; CHECK-NEXT:    WIDEN ir<[[X:%.+]]> = load ir<%a>
; CHECK-NEXT:    WIDEN ir<[[Y:%.+]]> = load ir<%b>
; CHECK-NEXT:    WIDEN ir<[[S:%.+]]> = add nsw ir<[[X]]>, ir<[[Y]]>
; CHECK-NEXT:    WIDEN store ir<%c>, ir<[[S]]>
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
;
  %a1 = getelementptr i32, ptr %a, i64 1
  %b1 = getelementptr i32, ptr %b, i64 1
  %c1 = getelementptr i32, ptr %c, i64 1
  %x0 = load i32, ptr %a
  %x1 = load i32, ptr %a1
  %y0 = load i32, ptr %b
  %y1 = load i32, ptr %b1
  %s0 = add nsw i32 %x0, %y0
  %s1 = add nsw i32 %x1, %y1
  store i32 %s0, ptr %c
  store i32 %s1, ptr %c1
  ret void
}

define void @print_vplan_cast(ptr %a, ptr %c) {
; CHECK-LABEL: SLP: VPlan for tree:
; CHECK-NEXT:  VPlan 'SLP tree for UF>=1' {
; CHECK-EMPTY:
; CHECK-NEXT:  {{.+}}:
; CHECK-NEXT:    WIDEN ir<[[X:%.+]]> = load ir<%a>
; CHECK-NEXT:    WIDEN-CAST ir<[[Z:%.+]]> = zext ir<[[X]]> to i32
; CHECK-NEXT:    WIDEN store ir<%c>, ir<[[Z]]>
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
;
  %a1 = getelementptr i16, ptr %a, i64 1
  %c1 = getelementptr i32, ptr %c, i64 1
  %x0 = load i16, ptr %a
  %x1 = load i16, ptr %a1
  %z0 = zext i16 %x0 to i32
  %z1 = zext i16 %x1 to i32
  store i32 %z0, ptr %c
  store i32 %z1, ptr %c1
  ret void
}
