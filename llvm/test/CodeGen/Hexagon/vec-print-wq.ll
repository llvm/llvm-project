; RUN: llc -mtriple=hexagon -mattr=+hvxv60,+hvx-length128b \
; RUN:   -enable-hexagon-vector-print < %s | FileCheck %s

; Test coverage for HexagonVectorPrint: exercise the W (double vector)
; and V (single vector) register printing paths.

; CHECK-LABEL: test_vecprint_v:
; CHECK: InlineAsm Start
; CHECK: .word
; CHECK: InlineAsm End
define void @test_vecprint_v(<32 x i32> %a, <32 x i32> %b, ptr %p) #0 {
entry:
  %add = add <32 x i32> %a, %b
  store <32 x i32> %add, ptr %p, align 128
  ret void
}

; The W register case should emit two inline asm blocks (one for each V sub-reg).
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

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
