; RUN: opt < %s -passes="print<inline-cost>" 2>&1 | FileCheck %s

; CHECK:       Analyzing call of callee1... (caller:foo)
; CHECK-NEXT: define i32 @callee1(i32 %x) {
; CHECK-NEXT: cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}
; CHECK-NEXT:   %x1 = add i32 %x, 1
; CHECK-NEXT: cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}
; CHECK-NEXT:   %x2 = add i32 %x1, 1
; CHECK-NEXT: cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}
; CHECK-NEXT:   %x3 = add i32 %x2, 1
; CHECK-NEXT: cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}
; CHECK-NEXT:   ret i32 %x3
; CHECK-NEXT: }

define i32 @foo(i32 %y) {
  %x = call i32 @callee1(i32 %y)
  ret i32 %x
}

define i32 @callee1(i32 %x) {
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1
  ret i32 %x3
}
