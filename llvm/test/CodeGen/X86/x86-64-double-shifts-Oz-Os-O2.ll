; RUN: llc < %s -march=x86-64 -mcpu=bdver1 | FileCheck %s

; clang -Oz -c test1.cpp -emit-llvm -S -o
; Verify that we generate shld insruction when we are optimizing for size,
; even for X86_64 processors that are known to have poor latency double 
; precision shift instructions.
; uint64_t lshift10(uint64_t a, uint64_t b)
; {
;     return (a << 10) | (b >> 54);
; }

; Function Attrs: minsize nounwind readnone uwtable
define i64 @_Z8lshift10mm(i64 %a, i64 %b) #0 {
entry:
; CHECK:   shldq   $10
  %shl = shl i64 %a, 10
  %shr = lshr i64 %b, 54
  %or = or i64 %shr, %shl
  ret i64 %or
}

attributes #0 = { minsize nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }


; clang -Os -c test2.cpp -emit-llvm -S
; Verify that we generate shld insruction when we are optimizing for size,
; even for X86_64 processors that are known to have poor latency double
; precision shift instructions.
; uint64_t lshift11(uint64_t a, uint64_t b)
; {
;     return (a << 11) | (b >> 53);
; }

; Function Attrs: nounwind optsize readnone uwtable
define i64 @_Z8lshift11mm(i64 %a, i64 %b) #1 {
entry:
; CHECK:   shldq   $11
  %shl = shl i64 %a, 11
  %shr = lshr i64 %b, 53
  %or = or i64 %shr, %shl
  ret i64 %or
}

attributes #1 = { nounwind optsize readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

; clang -O2 -c test2.cpp -emit-llvm -S
; Verify that we do not generate shld insruction when we are not optimizing
; for size for X86_64 processors that are known to have poor latency double
; precision shift instructions.
; uint64_t lshift12(uint64_t a, uint64_t b)
; {
;     return (a << 12) | (b >> 52);
; }

; Function Attrs: nounwind optsize readnone uwtable
define i64 @_Z8lshift12mm(i64 %a, i64 %b) #2 {
entry:
; CHECK:       shlq    $12
; CHECK-NEXT:  shrq    $52
  %shl = shl i64 %a, 12
  %shr = lshr i64 %b, 52
  %or = or i64 %shr, %shl
  ret i64 %or
}

attributes #2= { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

