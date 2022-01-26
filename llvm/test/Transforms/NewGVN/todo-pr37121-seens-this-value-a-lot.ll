; REQUIRES: asserts

; RUN: opt -passes=newgvn -S %s | FileCheck %s

; XFAIL: *

; TODO: Current NewGVN crashes on the function below. See PR37121.

define hidden void @foo() {
top:
  %.promoted = load i8, i8* undef, align 8
  br label %if

;; This is really a multi-valued phi, because the phi is defined by an expression of the phi.
;; This means that we can't propagate the value over the backedge, because we'll just cycle
;; through every value.

if:                                               ; preds = %if, %top
  %0 = phi i8 [ %1, %if ], [ %.promoted, %top ]
  %1 = xor i8 %0, undef
  br i1 false, label %L50, label %if

L50:                                              ; preds = %if
  %.lcssa = phi i8 [ %1, %if ]
  store i8 %.lcssa, i8* undef, align 8
  ret void
}
