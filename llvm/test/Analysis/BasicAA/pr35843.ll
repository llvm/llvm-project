; RUN: opt %s -passes=aa-eval -disable-output 2>&1 | FileCheck %s

; CHECK: 6 Total Alias Queries Performed
; CHECK-NEXT: 6 no alias responses

define void @patatino() {
BB:
  %G22 = getelementptr i1*, i1** undef, i8 -1
  %B1 = mul i66 undef, 9223372036854775808
  %G45 = getelementptr i1**, i1*** undef, i66 %B1
  ret void
}
