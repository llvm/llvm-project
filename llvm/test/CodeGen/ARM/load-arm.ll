; RUN: llc -mtriple=arm %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7 %s -o - | FileCheck %s

; We ended up feeding a deleted node back to TableGen when we converted "Off *
; 410" into "(Off * 205) << 1", where the multiplication already existed in the
; DAG.

; CHECK-LABEL: addrmode_cse_mutation:
; CHECK: {{mul|muls}}    [[OFFSET:r[0-9]+]], {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: {{ldrb|ldrb.w}} {{r[0-9]+}}, [r0, [[OFFSET]], lsl #3]
define i32 @addrmode_cse_mutation(i8* %base, i32 %count) {
  %offset = mul i32 %count, 277288
  %ptr = getelementptr i8, i8* %base, i32 %offset
  %val = load volatile i8, i8* %ptr
  %res = mul i32 %count, 34661
  ret i32 %res
}
