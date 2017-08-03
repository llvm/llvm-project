; RUN: llc < %s -march=x86

; rdar://7983260

%struct.T0 = type {}

define void @fn4(%struct.T0* byval %arg0) nounwind ssp {
entry:
  ret void
}
