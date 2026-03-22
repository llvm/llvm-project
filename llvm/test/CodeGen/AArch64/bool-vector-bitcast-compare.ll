; RUN: llc -global-isel=false -mtriple=aarch64-linux-gnu -o - %s | FileCheck %s

define i64 @match_any_byte(<8 x i8> %haystack, i8 %needle) {
; CHECK-LABEL: match_any_byte:
; CHECK:       // %bb.0:                               // %bb1
; CHECK-NEXT:    dup     v1.8b, w0
; CHECK-NEXT:    mov     w8, #999                        // =0x3e7
; CHECK-NEXT:    cmeq    v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    fmov    x9, d0
; CHECK-NEXT:    cmp     x9, #0
; CHECK-NEXT:    mov     w9, #777                        // =0x309
; CHECK-NEXT:    csel    x0, x9, x8, ne
; CHECK-NEXT:    ret
bb1:
  %0 = insertelement <8 x i8> poison, i8 %needle, i64 0
  %1 = shufflevector <8 x i8> %0, <8 x i8> poison, <8 x i32> zeroinitializer
  %2 = icmp eq <8 x i8> %haystack, %1
  %3 = bitcast <8 x i1> %2 to i8
  %4 = icmp ne i8 %3, 0
  %5 = select i1 %4, i64 777, i64 999
  ret i64 %5
}

define i64 @match_all_byte(<8 x i8> %haystack, i8 %needle) {
; CHECK-LABEL: match_all_byte:
; CHECK:       // %bb.0
; CHECK-NEXT:    dup     v1.8b, w0
; CHECK-NEXT:    mov     w8, #999                        // =0x3e7
; CHECK-NEXT:    cmeq    v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    fmov    x9, d0
; CHECK-NEXT:    cmn     x9, #1
; CHECK-NEXT:    mov     w9, #777                        // =0x309
; CHECK-NEXT:    csel    x0, x9, x8, eq
; CHECK-NEXT:    ret
bb1:
  %0 = insertelement <8 x i8> poison, i8 %needle, i64 0
  %1 = shufflevector <8 x i8> %0, <8 x i8> poison, <8 x i32> zeroinitializer
  %2 = icmp eq <8 x i8> %haystack, %1
  %3 = bitcast <8 x i1> %2 to i8
  %4 = icmp eq i8 %3, -1
  %5 = select i1 %4, i64 777, i64 999
  ret i64 %5
}

define i1 @match_any_bool(<8 x i8> %haystack, i8 %needle) {
; CHECK-LABEL: match_any_bool:
; CHECK:       // %bb.0
; CHECK-NEXT:    dup     v1.8b, w0
; CHECK-NEXT:    cmeq    v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    fmov    x8, d0
; CHECK-NEXT:    cmp     x8, #0
; CHECK-NEXT:    cset    w0, ne
; CHECK-NEXT:    ret
bb1:
  %0 = insertelement <8 x i8> poison, i8 %needle, i64 0
  %1 = shufflevector <8 x i8> %0, <8 x i8> poison, <8 x i32> zeroinitializer
  %2 = icmp eq <8 x i8> %haystack, %1
  %3 = bitcast <8 x i1> %2 to i8
  %4 = icmp ne i8 %3, 0
  ret i1 %4
}

define i64 @match_any_byte_16(<16 x i8> %haystack, i8 %needle) {
; CHECK-LABEL: match_any_byte_16:
; CHECK:       // %bb.0
; CHECK-NEXT:    dup     v1.16b, w0
; CHECK-NEXT:    mov     w8, #999                        // =0x3e7
; CHECK-NEXT:    cmeq    v0.16b, v0.16b, v1.16b
; CHECK-NEXT:    ext     v1.16b, v0.16b, v0.16b, #8
; CHECK-NEXT:    orr     v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    fmov    x9, d0
; CHECK-NEXT:    cmp     x9, #0
; CHECK-NEXT:    mov     w9, #777                        // =0x309
; CHECK-NEXT:    csel    x0, x9, x8, ne
; CHECK-NEXT:    ret
bb1:
  %0 = insertelement <16 x i8> poison, i8 %needle, i64 0
  %1 = shufflevector <16 x i8> %0, <16 x i8> poison, <16 x i32> zeroinitializer
  %2 = icmp eq <16 x i8> %haystack, %1
  %3 = bitcast <16 x i1> %2 to i16
  %4 = icmp ne i16 %3, 0
  %5 = select i1 %4, i64 777, i64 999
  ret i64 %5
}
