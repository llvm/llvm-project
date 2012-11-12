; Test that the EHABI unwind instruction generator does not encounter any
; unfamiliar instructions.
; RUN: llc < %s -mtriple=thumbv7 -arm-enable-ehabi -disable-fp-elim
; RUN: llc < %s -mtriple=thumbv7 -arm-enable-ehabi
; RUN: llc < %s -mtriple=thumbv7 -arm-enable-ehabi -arm-enable-ehabi-descriptors

define void @_Z1fv() nounwind {
entry:
  ret void
}

define void @_Z1gv() nounwind {
entry:
  call void @_Z1fv()
  ret void
}
