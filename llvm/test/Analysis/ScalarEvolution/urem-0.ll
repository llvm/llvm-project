; RUN: opt < %s "-passes=print<scalar-evolution>" -disable-output 2>&1 | FileCheck %s

define i8 @foo(i8 %a) {
; CHECK-LABEL: @foo
        %t0 = urem i8 %a, 27
; CHECK: %t0 = urem i8 %a, 27
; CHECK-NEXT: -->  ((-27 * (%a /u 27)) + %a)
        ret i8 %t0
}

define i8 @bar(i8 %a) {
; CHECK-LABEL: @bar
        %t1 = urem i8 %a, 1
; CHECK: %t1 = urem i8 %a, 1
; CHECK-NEXT: -->  0
        ret i8 %t1
}

define i8 @baz(i8 %a) {
; CHECK-LABEL: @baz
        %t2 = urem i8 %a, 32
; CHECK: %t2 = urem i8 %a, 32
; CHECK-NEXT: -->  (zext i5 (trunc i8 %a to i5) to i8)
        ret i8 %t2
}

define i8 @qux(i8 %a) {
; CHECK-LABEL: @qux
        %t3 = urem i8 %a, 2
; CHECK: %t3 = urem i8 %a, 2
; CHECK-NEXT: -->  (zext i1 (trunc i8 %a to i1) to i8)
        ret i8 %t3
}
