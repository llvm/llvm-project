; RUN: llc -march=hexagon < %s | FileCheck %s

; DAG combiner folds sequences of shifts, which can sometimes obscure
; optimization opportunities. For example
;
;   unsigned int c(unsigned int b, unsigned int *a) {
;     unsigned int bitidx = b >> 5;
;     return a[bitidx];
;   }
;
; produces
;   (add x (shl (srl y 5) 2))
; which is then folded into
;   (add x (and (srl y 3) 1FFFFFFC))
;
; That results in a constant-extended and:
;   r0 = and(##536870908,lsr(r0,#3))
;   r0 = memw(r1+r0<<#0)
; whereas
;   r0 = lsr(r0,#5)
;   r0 = memw(r1+r0<<#2)
; is more desirable.

target triple = "hexagon"

; CHECK-LABEL: load_0
; CHECK: memw(r{{[0-9]+}}+r{{[0-9]}}<<#2)
define i32 @load_0(i32 %b, ptr nocapture readonly %a) #0 {
entry:
  %shr = lshr i32 %b, 5
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %shr
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

; This would require r0<<#3, which is not legal.
; CHECK-LABEL: load_1
; CHECK: memw(r{{[0-9]+}}+r{{[0-9]}}<<#0)
define i32 @load_1(i32 %b, ptr nocapture readonly %a) #0 {
entry:
  %shr = lshr i32 %b, 5
  %arrayidx = getelementptr inbounds [3 x i32], ptr %a, i32 %shr, i32 0
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

; CHECK-LABEL: store_0
; CHECK: memw(r{{[0-9]+}}+r{{[0-9]}}<<#2)
define void @store_0(i32 %b, ptr nocapture %a, i32 %v) #1 {
entry:
  %shr = lshr i32 %b, 5
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %shr
  store i32 %v, ptr %arrayidx, align 4
  ret void
}

attributes #0 = { norecurse nounwind readonly "target-cpu"="hexagonv60" "target-features"="-hvx,-long-calls" }
attributes #1 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="-hvx,-long-calls" }

