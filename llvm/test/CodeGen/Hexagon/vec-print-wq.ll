; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -mattr=+hvxv68,+hvx-length128b \
; RUN:   -enable-hexagon-vector-print < %s | FileCheck %s

; Test coverage for HexagonVectorPrint: exercise the W (double vector)
; and V (single vector) register printing paths.

; Use two stores so that the V result is not entirely consumed by a .new
; store, leaving a live V register for the vector-print pass to instrument.
; CHECK-LABEL: test_vecprint_v:
; CHECK: InlineAsm Start
; CHECK: .word
; CHECK: InlineAsm End
define void @test_vecprint_v(<32 x i32> %a, <32 x i32> %b, ptr %p, ptr %q) #0 {
entry:
  %add = add <32 x i32> %a, %b
  store <32 x i32> %add, ptr %p, align 128
  store <32 x i32> %add, ptr %q, align 128
  ret void
}

; The W register case should emit inline asm blocks for each V sub-register.
; CHECK-LABEL: test_vecprint_w:
; CHECK: InlineAsm Start
; CHECK: .word
; CHECK: InlineAsm End
; CHECK: InlineAsm Start
; CHECK: .word
; CHECK: InlineAsm End
define void @test_vecprint_w(<64 x i32> %a, <64 x i32> %b, ptr %p) #0 {
entry:
  %add = add <64 x i32> %a, %b
  store <64 x i32> %add, ptr %p, align 128
  ret void
}

attributes #0 = { nounwind }
