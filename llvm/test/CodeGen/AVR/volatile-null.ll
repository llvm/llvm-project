; RUN: llc < %s -mtriple=avr | FileCheck %s

define i8 @load_volatile_null() {
; CHECK-LABEL: load_volatile_null:
; CHECK: lds r24, 0
    %result = load volatile i8, ptr null
    ret i8 %result
}

define void @store_volatile_null(i8 %a) {
; CHECK-LABEL: store_volatile_null:
; CHECK: sts 0, r24
    store volatile i8 %a, ptr null
    ret void
}
