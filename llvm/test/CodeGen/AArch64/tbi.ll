; RUN: llc -aarch64-use-tbi -mtriple=arm64-apple-ios8.0.0 < %s \
; RUN:     | FileCheck --check-prefix=TBI    --check-prefix=BOTH %s
; RUN: llc -aarch64-use-tbi -mtriple=arm64-apple-ios7.1.0 < %s \
; RUN:     | FileCheck --check-prefix=NO_TBI --check-prefix=BOTH %s

; BOTH-LABEL:ld_and32:
; TBI-NOT: and x
; NO_TBI: and x
define i32 @ld_and32(i64 %p) {
  %and = and i64 %p, 72057594037927935
  %cast = inttoptr i64 %and to i32*
  %load = load i32, i32* %cast
  ret i32 %load
}

; load (r & MASK) + 4
; BOTH-LABEL:ld_and_plus_offset:
; TBI-NOT: and x
; NO_TBI: and x
define i32 @ld_and_plus_offset(i64 %p) {
  %and = and i64 %p, 72057594037927935
  %cast = inttoptr i64 %and to i32*
  %gep = getelementptr i32, i32* %cast, i64 4
  %load = load i32, i32* %gep
  ret i32 %load
}

; load (r & WIDER_MASK)
; BOTH-LABEL:ld_and32_wider:
; TBI-NOT: and x
; NO_TBI: and x
define i32 @ld_and32_wider(i64 %p) {
  %and = and i64 %p, 1152921504606846975
  %cast = inttoptr i64 %and to i32*
  %load = load i32, i32* %cast
  ret i32 %load
}

; BOTH-LABEL:ld_and64:
; TBI-NOT: and x
; NO_TBI: and x
define i64 @ld_and64(i64 %p) {
  %and = and i64 %p, 72057594037927935
  %cast = inttoptr i64 %and to i64*
  %load = load i64, i64* %cast
  ret i64 %load
}

; BOTH-LABEL:st_and32:
; TBI-NOT: and x
; NO_TBI: and x
define void @st_and32(i64 %p, i32 %v) {
  %and = and i64 %p, 72057594037927935
  %cast = inttoptr i64 %and to i32*
  store i32 %v, i32* %cast
  ret void
}

; load (x1 + x2) & MASK
; BOTH-LABEL:ld_ro:
; TBI-NOT: and x
; NO_TBI: and x
define i32 @ld_ro(i64 %a, i64 %b) {
  %p = add i64 %a, %b
  %and = and i64 %p, 72057594037927935
  %cast = inttoptr i64 %and to i32*
  %load = load i32, i32* %cast
  ret i32 %load
}

; load (r1 & MASK) + r2
; BOTH-LABEL:ld_ro2:
; TBI-NOT: and x
; NO_TBI: and x
define i32 @ld_ro2(i64 %a, i64 %b) {
  %and = and i64 %a, 72057594037927935
  %p = add i64 %and, %b
  %cast = inttoptr i64 %p to i32*
  %load = load i32, i32* %cast
  ret i32 %load
}

; load (r1 & MASK) | r2
; BOTH-LABEL:ld_indirect_and:
; TBI-NOT: and x
; NO_TBI: and x
define i32 @ld_indirect_and(i64 %r1, i64 %r2) {
  %and = and i64 %r1, 72057594037927935
  %p = or i64 %and, %r2
  %cast = inttoptr i64 %p to i32*
  %load = load i32, i32* %cast
  ret i32 %load
}

; BOTH-LABEL:ld_and32_narrower:
; BOTH: and x
define i32 @ld_and32_narrower(i64 %p) {
  %and = and i64 %p, 36028797018963967
  %cast = inttoptr i64 %and to i32*
  %load = load i32, i32* %cast
  ret i32 %load
}

; BOTH-LABEL:ld_and8:
; BOTH: and x
define i32 @ld_and8(i64 %base, i8 %off) {
  %off_masked = and i8 %off, 63
  %off_64 = zext i8 %off_masked to i64
  %p = add i64 %base, %off_64
  %cast = inttoptr i64 %p to i32*
  %load = load i32, i32* %cast
  ret i32 %load
}
