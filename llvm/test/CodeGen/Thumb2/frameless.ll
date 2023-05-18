; RUN: llc < %s -mtriple=thumbv7-apple-darwin -frame-pointer=all | not grep mov
; RUN: llc < %s -mtriple=thumbv7-linux -frame-pointer=all | not grep mov

define void @t() nounwind readnone {
  ret void
}
