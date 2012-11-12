; RUN: llc -O2 -print-after-all < %s 2>/dev/null

define void @tester(){
  ret void
}

