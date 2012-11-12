; Ensure that all immediate values that are 32-bits or less can be loaded 
; using a single instruction and that immediate values 64-bits or less can
; be loaded using two instructions.
;
; RUN: llc < %s -march=mblaze | FileCheck %s
; RUN: llc < %s -march=mblaze -mattr=+fpu | FileCheck -check-prefix=FPU %s

define i8 @retimm_i8() {
    ; CHECK:        retimm_i8:
    ; CHECK:        rtsd
    ; CHECK-NEXT:   add
    ; FPU:          retimm_i8:
    ; FPU:          rtsd
    ; FPU-NEXT:     add
    ret i8 123
}

define i16 @retimm_i16() {
    ; CHECK:        retimm_i16:
    ; CHECK:        rtsd
    ; CHECK-NEXT:   add
    ; FPU:          retimm_i16:
    ; FPU:          rtsd
    ; FPU-NEXT:     add
    ret i16 31212
}

define i32 @retimm_i32() {
    ; CHECK:        retimm_i32:
    ; CHECK:        add
    ; CHECK-NEXT:   rtsd
    ; FPU:          retimm_i32:
    ; FPU:          add
    ; FPU-NEXT:     rtsd
    ret i32 2938128
}

define i64 @retimm_i64() {
    ; CHECK:        retimm_i64:
    ; CHECK:        add
    ; CHECK-NEXT:   rtsd
    ; CHECK-NEXT:   add
    ; FPU:          retimm_i64:
    ; FPU:          add
    ; FPU-NEXT:     rtsd
    ; FPU-NEXT:     add
    ret i64 94581823
}

define float @retimm_float() {
    ; CHECK:        retimm_float:
    ; CHECK:        add
    ; CHECK-NEXT:   rtsd
    ; FPU:          retimm_float:
    ; FPU:          or
    ; FPU-NEXT:     rtsd
    ret float 12.0
}

define double @retimm_double() {
    ; CHECK:        retimm_double:
    ; CHECK:        add
    ; CHECK-NEXT:   add
    ; CHECK-NEXT:   rtsd
    ; FPU:          retimm_double:
    ; FPU:          add
    ; FPU-NEXT:     add
    ; FPU-NEXT:     rtsd
    ret double 598382.39283873
}
