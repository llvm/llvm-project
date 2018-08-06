; RUN: opt %s -aa-eval -disable-output 2>&1 | FileCheck %s

; CHECK: 6 Total Alias Queries Performed
; CHECK-NEXT: 6 no alias responses

define void @patatino() {
  %G26 = getelementptr i1, i1* undef, i1 undef
  %B20 = shl i8 -128, 16
  %G47 = getelementptr i1*, i1** undef, i8 %B20
  ret void
}
