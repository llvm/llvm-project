; RUN: llvm-link -S %s %p/Inputs/type-unique-unrelated2.ll %p/Inputs/type-unique-unrelated3.ll | FileCheck %s

; CHECK: %t = type { ptr }

; CHECK: define %t @f2() {
; CHECK-NEXT:   %x = call %t @f2()
; CHECK-NEXT:   ret %t %x
; CHECK-NEXT: }

; CHECK: define %t @g2() {
; CHECK-NEXT:   %x = call %t @g()
; CHECK-NEXT:   ret %t %x
; CHECK-NEXT: }

; CHECK: define %t @g() {
; CHECK-NEXT:  %x = call %t @f()
; CHECK-NEXT:  ret %t %x
; CHECK-NEXT: }

; The idea of this test is that the %t in this file and the one in
; type-unique-unrelated2.ll look unrelated until type-unique-unrelated3.ll
; is merged in.

%t = type { ptr }
declare %t @f()

define %t @f2() {
 %x = call %t @f2()
 ret %t %x
}

