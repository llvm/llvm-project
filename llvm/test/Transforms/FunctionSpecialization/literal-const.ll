; RUN: opt -S --passes="ipsccp<func-spec>" \
; RUN:        -force-specialization < %s | FileCheck %s -check-prefix CHECK-NOLIT
; RUN: opt -S --passes="ipsccp<func-spec>" \
; RUN:        -funcspec-for-literal-constant \
; RUN:        -force-specialization < %s | FileCheck %s -check-prefix CHECK-LIT

define i32 @f0(i32 noundef %x) {
entry:
  %call = tail call i32 @neg(i32 noundef %x, i1 noundef zeroext false)
  ret i32 %call
}

define i32 @f1(i32 noundef %x) {
entry:
  %call = tail call i32 @neg(i32 noundef %x, i1 noundef zeroext true)
  ret i32 %call
}

define i32 @g0(i32 noundef %x) {
entry:
  %call = tail call i32 @add(i32 noundef %x, i32 noundef 1)
  ret i32 %call
}

define i32 @g1(i32 noundef %x) {
entry:
  %call = tail call i32 @add(i32 noundef %x, i32 noundef 2)
  ret i32 %call
}

define float @h0(float noundef %x) {
entry:
  %call = tail call float @addf(float noundef %x, float noundef 1.000000e+00)
  ret float %call
}

define float @h1(float noundef %x) {
entry:
  %call = tail call float @addf(float noundef %x, float noundef 2.000000e+00)
  ret float %call
}

define internal i32 @neg(i32 noundef %x, i1 noundef zeroext %b) {
entry:
  %sub = sub nsw i32 0, %x
  %cond = select i1 %b, i32 %sub, i32 %x
  ret i32 %cond
}

define internal i32 @add(i32 noundef %x, i32 noundef %y) {
entry:
  %add = add nsw i32 %y, %x
  ret i32 %add
}

define internal float @addf(float noundef %x, float noundef %y) {
entry:
  %add = fadd float %x, %y
  ret float %add
}


; Check no functions were specialised.
; CHECK-NOLIT-NOT: @neg.
; CHECK-NOLIT-NOT: @add.
; CHECK-NOLIT-NOT: @addf.

; CHECK-LIT-LABEL: define i32 @f0
; CHECK-LIT: call i32 @neg.[[#A:]]

; CHECK-LIT-LABEL: define i32 @f1
; CHECK-LIT: call i32 @neg.[[#B:]]

; CHECK-LIT-LABEL: define i32 @g0
; CHECK-LIT: call i32 @add.[[#C:]]

; CHECK-LIT-LABEL: define i32 @g1
; CHECK-LIT: call i32 @add.[[#D:]]

; CHECK-LIT-LABEL: define float @h0
; CHECK-LIT: call float @addf.[[#E:]]

; CHECK-LIT-LABEL: define float @h1
; CHECK-LIT: call float @addf.[[#F:]]

; Check all of `neg`, `add`, and `addf` were specialised.
; CHECK-LIT-DAG: @neg.[[#A]]
; CHECK-LIT-DAG: @neg.[[#B]]
; CHECK-LIT-DAG: @add.[[#C]]
; CHECK-LIT-DAG: @add.[[#D]]
; CHECK-LIT-DAG: @addf.[[#E]]
; CHECK-LIT-DAG: @addf.[[#F]]
