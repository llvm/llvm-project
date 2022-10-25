; RUN: opt -S --passes=function-specialization \
; RUN:        -force-function-specialization < %s | FileCheck %s -check-prefix CHECK-NOLIT
; RUN: opt -S --passes=function-specialization \
; RUN:        -function-specialization-for-literal-constant \
; RUN:        -force-function-specialization < %s | FileCheck %s -check-prefix CHECK-LIT

define dso_local i32 @f0(i32 noundef %x) {
entry:
  %call = tail call fastcc i32 @neg(i32 noundef %x, i1 noundef zeroext false)
  ret i32 %call
}

define internal fastcc i32 @neg(i32 noundef %x, i1 noundef zeroext %b) {
entry:
  %sub = sub nsw i32 0, %x
  %cond = select i1 %b, i32 %sub, i32 %x
  ret i32 %cond
}

define dso_local i32 @f1(i32 noundef %x) {
entry:
  %call = tail call fastcc i32 @neg(i32 noundef %x, i1 noundef zeroext true)
  ret i32 %call
}

define dso_local i32 @g0(i32 noundef %x) {
entry:
  %call = tail call fastcc i32 @add(i32 noundef %x, i32 noundef 1)
  ret i32 %call
}

define internal fastcc i32 @add(i32 noundef %x, i32 noundef %y) {
entry:
  %add = add nsw i32 %y, %x
  ret i32 %add
}

define dso_local i32 @g1(i32 noundef %x) {
entry:
  %call = tail call fastcc i32 @add(i32 noundef %x, i32 noundef 2)
  ret i32 %call
}

define dso_local float @h0(float noundef %x) {
entry:
  %call = tail call fastcc float @addf(float noundef %x, float noundef 1.000000e+00)
  ret float %call
}

define internal fastcc float @addf(float noundef %x, float noundef %y) {
entry:
  %add = fadd float %x, %y
  ret float %add
}

define dso_local float @h1(float noundef %x) {
entry:
  %call = tail call fastcc float @addf(float noundef %x, float noundef 2.000000e+00)
  ret float %call
}

; Check no functions were specialised.
; CHECK-NOLIT-NOT: @neg.
; CHECK-NOLIT-NOT: @add.
; CHECK-NOLIT-NOT: @addf.

; Check all of `neg`, `add`, and `addf` were specialised.
; CHECK-LIT-DAG: @neg.1
; CHECK-LIT-DAG: @neg.2
; CHECK-LIT-DAG: @add.3
; CHECK-LIT-DAG: @add.4
; CHECK-LIT-DAG: @addf.5
; CHECK-LIT-DAG: @addf.6
