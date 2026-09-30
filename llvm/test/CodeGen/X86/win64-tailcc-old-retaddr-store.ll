; RUN: llc -mtriple=x86_64-pc-windows-msvc -stop-after=finalize-isel < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-linux-gnu -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=LINUX

; The call grows the stack-argument area, so an outgoing argument overwrites the
; caller's return-address slot. On Win64 the value stored there is kept live in
; a vreg that the tail call also uses, so the store can later be sunk to just
; before the epilogue. The store is volatile so it stays ordered after the
; store of the new return address.

declare tailcc void @g(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define tailcc void @f(i64 %a, i64 %b) {
; CHECK-LABEL: name: f
; CHECK:         MOV64mr %fixed-stack.{{[0-9]+}}, 1, $noreg, 0, $noreg, %[[V:[0-9]+]] :: (volatile store (s64) into %fixed-stack.{{[0-9]+}})
; CHECK:         TCRETURNdi64 @g, {{.*}}implicit %[[V]]
;
; LINUX-LABEL: name: f
; LINUX:         TCRETURNdi64 {{.*}}
; LINUX-NOT:     implicit %
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
}
