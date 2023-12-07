; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}


; CHECK: bfe0
define i32 @bfe0(i32 %a) {
; CHECK: bfe.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, 4, 4
; CHECK-NOT: shr
; CHECK-NOT: and
  %val0 = ashr i32 %a, 4
  %val1 = and i32 %val0, 15
  ret i32 %val1
}

; CHECK: bfe1
define i32 @bfe1(i32 %a) {
; CHECK: bfe.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, 3, 3
; CHECK-NOT: shr
; CHECK-NOT: and
  %val0 = ashr i32 %a, 3
  %val1 = and i32 %val0, 7
  ret i32 %val1
}

; CHECK: bfe2
define i32 @bfe2(i32 %a) {
; CHECK: bfe.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, 5, 3
; CHECK-NOT: shr
; CHECK-NOT: and
  %val0 = ashr i32 %a, 5
  %val1 = and i32 %val0, 7
  ret i32 %val1
}

; CHECK-LABEL: no_bfe_on_32bit_overflow
define i32 @no_bfe_on_32bit_overflow(i32 %a) {
; CHECK-NOT: bfe.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, 31, 4
  %val0 = ashr i32 %a, 31
  %val1 = and i32 %val0, 15
  ret i32 %val1
}

; CHECK-LABEL: no_bfe_on_32bit_overflow_shr_and_pair
define i32 @no_bfe_on_32bit_overflow_shr_and_pair(i32 %a) {
; CHECK: shr.s32 %r{{[0-9]+}}, %r{{[0-9]+}}, 31
; CHECK: and.b32 %r{{[0-9]+}}, %r{{[0-9]+}}, 15
  %val0 = ashr i32 %a, 31
  %val1 = and i32 %val0, 15
  ret i32 %val1
}

; CHECK-LABEL: no_bfe_on_64bit_overflow
define i64 @no_bfe_on_64bit_overflow(i64 %a) {
; CHECK-NOT: bfe.u64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, 63, 3
  %val0 = ashr i64 %a, 63
  %val1 = and i64 %val0, 7
  ret i64 %val1
}

; CHECK-LABEL: no_bfe_on_64bit_overflow_shr_and_pair
define i64 @no_bfe_on_64bit_overflow_shr_and_pair(i64 %a) {
; CHECK: shr.s64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, 63
; CHECK: and.b64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, 7
  %val0 = ashr i64 %a, 63
  %val1 = and i64 %val0, 7
  ret i64 %val1
}
