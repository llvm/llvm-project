; RUN: llc < %s -march=x86-64 -mcpu=bdver1 | FileCheck %s
; Verify that for the architectures that are known to have poor latency
; double precision shift instructions we generate alternative sequence 
; of instructions with lower latencies instead of shld instruction.

;uint64_t lshift1(uint64_t a, uint64_t b)
;{
;    return (a << 1) | (b >> 63);
;}

; CHECK:             lshift1:
; CHECK:             addq    {{.*}},{{.*}}
; CHECK-NEXT:        shrq    $63, {{.*}}
; CHECK-NEXT:        leaq    ({{.*}},{{.*}}), {{.*}}


define i64 @lshift1(i64 %a, i64 %b) nounwind readnone uwtable {
entry:
  %shl = shl i64 %a, 1
  %shr = lshr i64 %b, 63
  %or = or i64 %shr, %shl
  ret i64 %or
}

;uint64_t lshift2(uint64_t a, uint64_t b)
;{
;    return (a << 2) | (b >> 62);
;}

; CHECK:             lshift2:
; CHECK:             shlq    $2, {{.*}}
; CHECK-NEXT:        shrq    $62, {{.*}}
; CHECK-NEXT:        leaq    ({{.*}},{{.*}}), {{.*}}

define i64 @lshift2(i64 %a, i64 %b) nounwind readnone uwtable {
entry:
  %shl = shl i64 %a, 2
  %shr = lshr i64 %b, 62
  %or = or i64 %shr, %shl
  ret i64 %or
}

;uint64_t lshift7(uint64_t a, uint64_t b)
;{
;    return (a << 7) | (b >> 57);
;}

; CHECK:             lshift7:
; CHECK:             shlq    $7, {{.*}}
; CHECK-NEXT:        shrq    $57, {{.*}}
; CHECK-NEXT:        leaq    ({{.*}},{{.*}}), {{.*}}

define i64 @lshift7(i64 %a, i64 %b) nounwind readnone uwtable {
entry:
  %shl = shl i64 %a, 7
  %shr = lshr i64 %b, 57
  %or = or i64 %shr, %shl
  ret i64 %or
}

;uint64_t lshift63(uint64_t a, uint64_t b)
;{
;    return (a << 63) | (b >> 1);
;}

; CHECK:             lshift63:
; CHECK:             shlq    $63, {{.*}}
; CHECK-NEXT:        shrq    {{.*}}
; CHECK-NEXT:        leaq    ({{.*}},{{.*}}), {{.*}}

define i64 @lshift63(i64 %a, i64 %b) nounwind readnone uwtable {
entry:
  %shl = shl i64 %a, 63
  %shr = lshr i64 %b, 1
  %or = or i64 %shr, %shl
  ret i64 %or
}
