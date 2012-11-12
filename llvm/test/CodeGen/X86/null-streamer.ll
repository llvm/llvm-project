; Check the MCNullStreamer operates correctly, at least on a minimal test case.
;
; RUN: llc -filetype=null -o %t -march=x86 %s

define void @f0()  {
  ret void
}

define void @f1() {
  ret void
}
