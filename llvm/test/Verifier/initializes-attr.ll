; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Attribute 'initializes' requires interval lower less than upper
; CHECK-NEXT: ptr @lower_greater_than_upper1
define void @lower_greater_than_upper1(ptr initializes((4, 0)) %a) {
  ret void
}

; CHECK: Attribute 'initializes' requires interval lower less than upper
; CHECK-NEXT: ptr @lower_greater_than_upper2
define void @lower_greater_than_upper2(ptr initializes((0, 4), (8, 6)) %a) {
  ret void
}

; CHECK: Attribute 'initializes' requires intervals in ascending order!
; CHECK-NEXT: ptr @descending_order
define void @descending_order(ptr initializes((8, 12), (0, 4)) %a) {
  ret void
}

; CHECK: Attribute 'initializes' requires intervals merged!
; CHECK-NEXT: ptr @overlapping1
define void @overlapping1(ptr initializes((0, 4), (4, 8)) %a) {
  ret void
}

; CHECK: Attribute 'initializes' requires intervals merged!
; CHECK-NEXT: ptr @overlapping2
define void @overlapping2(ptr initializes((0, 4), (2, 8)) %a) {
  ret void
}
