; RUN: opt -analyze -scalar-evolution -scev-addops-inline-threshold=1 < %s | FileCheck --check-prefix=CHECK1 %s
; RUN: opt -analyze -scalar-evolution -scev-addops-inline-threshold=10 < %s | FileCheck --check-prefix=CHECK10 %s

define i32 @foo(i64 %p0, i32 %p1) {
; CHECK1: %add2 = add nsw i32 %mul1, %add
; CHECK1-NEXT: -->  ((trunc i64 %p0 to i32) * (1 + (trunc i64 %p0 to i32)) * (1 + %p1))

; CHECK10: %add2 = add nsw i32 %mul1, %add
; CHECK10-NEXT: -->  ((trunc i64 %p0 to i32) * (1 + ((trunc i64 %p0 to i32) * (1 + %p1)) + %p1))
entry:
  %tr = trunc i64 %p0 to i32
  %mul = mul nsw i32 %tr, %p1
  %add = add nsw i32 %mul, %tr
  %mul1 = mul nsw i32 %add, %tr
  %add2 = add nsw i32 %mul1, %add
  ret i32 %add2
}
