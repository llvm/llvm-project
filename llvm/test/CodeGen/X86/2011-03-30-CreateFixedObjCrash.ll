; RUN: llc < %s -mtriple=i686--

; rdar://7983260

%struct.T0 = type {}

define void @fn4(ptr byval(%struct.T0) %arg0) nounwind ssp {
entry:
  ret void
}
