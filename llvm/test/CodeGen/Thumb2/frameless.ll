; RUN: llc < %s -mtriple=thumbv7-apple-darwin -disable-fp-elim | not grep mov
; RUN: llc < %s -mtriple=thumbv7-linux -disable-fp-elim | not grep mov

define void @t() nounwind readnone {
  ret void
}
