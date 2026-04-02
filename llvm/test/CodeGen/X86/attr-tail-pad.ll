; RUN: llc -mtriple=x86_64-unknown-linux %s -o - | FileCheck %s

;; FIXME: Do we want .size to include the fill?

; CHECK:      a:                                      # @a
; CHECK:              retq
; CHECK-NEXT: .Lfunc_end0:
; CHECK-NEXT:         .size   a, .Lfunc_end0-a
; CHECK-NEXT:         .zero   (5-(.Lfunc_end0-a))&((5-(.Lfunc_end0-a))>=0),144

define hidden void @a() #0 {
entry:
  ret void
}

attributes #0 = { "tail-pad-to-size"="5" "tail-pad-value"="144" }
