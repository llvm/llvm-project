; RUN: not opt -verify < %s 2>&1 | FileCheck %s

; CHECK: Attribute 'inalloca' does not support unsized types!
; CHECK-NEXT: void (ptr)* @f
define void @f(ptr inalloca(token)) {
    ret void
}
