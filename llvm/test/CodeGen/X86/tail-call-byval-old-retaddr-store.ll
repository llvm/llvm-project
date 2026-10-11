; RUN: llc -mtriple=x86_64-linux-gnu -stop-after=finalize-isel < %s | FileCheck %s

; A byval argument whose destination covers the caller's return-address slot is
; split: the 8 bytes that land on that slot are copied with a volatile store, so
; they stay ordered after the store of the new return address, and the rest is
; copied normally.

declare tailcc void @h(ptr byval([9 x i64]) align 8)

define tailcc void @f(ptr %p) {
; CHECK-LABEL: name: f
; CHECK:         (volatile store (s64) into %fixed-stack.{{[0-9]+}})
; CHECK:         (volatile store (s64) into %fixed-stack.{{[0-9]+}} + 56)
; CHECK-NOT:     (store (s64) into %fixed-stack.{{[0-9]+}} + 56
; CHECK:         TCRETURNdi64
  musttail call tailcc void @h(ptr byval([9 x i64]) align 8 %p)
  ret void
}
