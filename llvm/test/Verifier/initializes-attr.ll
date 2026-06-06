; RUN: split-file %s %t
; RUN: not llvm-as < %s %t/lower_greater_than_upper1.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=Lower-GT-Upper1
; RUN: not llvm-as < %s %t/lower_greater_than_upper2.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=Lower-GT-Upper2
; RUN: not llvm-as < %s %t/descending_order.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DescOrder
; RUN: not llvm-as < %s %t/overlapping1.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=Overlapping1
; RUN: not llvm-as < %s %t/overlapping2.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=Overlapping2

;--- lower_greater_than_upper1.ll
; Lower-GT-Upper1: error: Invalid (unordered or overlapping) range list
define void @lower_greater_than_upper1(ptr initializes((4, 0)) %a) {
  ret void
}

;--- lower_greater_than_upper2.ll
; Lower-GT-Upper2: error: Invalid (unordered or overlapping) range list
define void @lower_greater_than_upper2(ptr initializes((0, 4), (8, 6)) %a) {
  ret void
}

;--- descending_order.ll
; DescOrder: error: Invalid (unordered or overlapping) range list
define void @descending_order(ptr initializes((8, 12), (0, 4)) %a) {
  ret void
}

;--- overlapping1.ll
; Overlapping1: error: Invalid (unordered or overlapping) range list
define void @overlapping1(ptr initializes((0, 4), (4, 8)) %a) {
  ret void
}

;--- overlapping2.ll
; Overlapping2: error: Invalid (unordered or overlapping) range list
define void @overlapping2(ptr initializes((0, 4), (2, 8)) %a) {
  ret void
}
