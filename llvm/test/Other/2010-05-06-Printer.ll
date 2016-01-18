; RUN: llc -O2 -print-after-all < %s 2>/dev/null
; REQUIRES: default_triple

define void @tester(){
  ret void
}

