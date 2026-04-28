; RUN: llc -mtriple=x86_64-unknown-linux %s -o - | FileCheck %s

; CHECK:      a:                                      # @a
; CHECK:              retq
; CHECK-NEXT: .Ltail_pad_start0:
; CHECK-NEXT:         .zero (5-(.Ltail_pad_start0-a))&((5-(.Ltail_pad_start0-a))>=0),144
; CHECK-NEXT: .Lfunc_end0:
; CHECK-NEXT:         .size   a, .Lfunc_end0-a

define hidden void @a() #0 {
entry:
  ret void
}

attributes #0 = { "tail-pad-to-size"="5" "tail-pad-value"="144" }
