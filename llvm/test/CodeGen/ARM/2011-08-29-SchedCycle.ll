; RUN: llc %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -o -

; When a i64 sub is expanded to subc + sube.
;   libcall #1
;      \
;       \        subc 
;        \       /  \
;         \     /    \
;          \   /    libcall #2
;           sube
;
; If the libcalls are not serialized (i.e. both have chains which are dag
; entry), legalizer can serialize them in arbitrary orders. If it's
; unlucky, it can force libcall #2 before libcall #1 in the above case.
;
;   subc
;    |
;   libcall #2
;    |
;   libcall #1
;    |
;   sube
;
; However since subc and sube are "glued" together, this ends up being a
; cycle when the scheduler combine subc and sube as a single scheduling
; unit.
;
; The right solution is to fix LegalizeType too chains the libcalls together.
; However, LegalizeType is not processing nodes in order. The fix now is to
; fix subc / sube (and addc / adde) to use physical register dependency instead.
; rdar://10019576

define void @t() nounwind {
entry:
  %tmp = load i64* undef, align 4
  %tmp5 = udiv i64 %tmp, 30
  %tmp13 = and i64 %tmp5, 64739244643450880
  %tmp16 = sub i64 0, %tmp13
  %tmp19 = and i64 %tmp16, 63
  %tmp20 = urem i64 %tmp19, 3
  %tmp22 = and i64 %tmp16, -272346829004752
  store i64 %tmp22, i64* undef, align 4
  store i64 %tmp20, i64* undef, align 4
  ret void
}
