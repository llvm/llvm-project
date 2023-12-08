; Test that the EHABI unwind instruction generator does not encounter any
; unfamiliar instructions.
; RUN: llc < %s -mtriple=thumbv7 -frame-pointer=all
; RUN: llc < %s -mtriple=thumbv7

define void @_Z1fv() nounwind {
entry:
  ret void
}

define void @_Z1gv() nounwind {
entry:
  call void @_Z1fv()
  ret void
}
