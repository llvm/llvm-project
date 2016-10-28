; RUN: opt -instcombine %s -S | FileCheck %s

; CHECK-LABEL: patatino
; CHECK-NEXT: ret <8 x i64*> undef
define <8 x i64*> @patatino() {
  %el = getelementptr i64, <8 x i64*> undef, <8 x i64> undef
  ret <8 x i64*> %el
}

; CHECK-LABEL: patatino2
; CHECK-NEXT: ret <8 x i64*> undef
define <8 x i64*> @patatino2() {
  %el = getelementptr inbounds i64, i64* undef, <8 x i64> undef
  ret <8 x i64*> %el
}
