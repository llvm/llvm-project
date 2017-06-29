; RUN: opt < %s -scalar-evolution -analyze | FileCheck %s

define i8 @foo(i8 %a) {
        %t0 = urem i8 %a, 27
; CHECK: %t0
; CHECK-NEXT: -->  ((-27 * (%a /u 27)) + %a)
        ret i8 %t0
}

define i8 @bar(i8 %a) {
        %t1 = urem i8 %a, 1
; CHECK: %t1
; CHECK-NEXT: -->  0
        ret i8 %t1
}
