; RUN: opt --disable-output --debug-pass-manager \
; RUN:   --passes='default<O2>' --disable-passes=early-cse,inline < %s 2>&1 | FileCheck %s
define void @test() {
  ret void
}

; CHECK: Skipping pass: InlinerPass on (test)
; CHECK: Skipping pass: EarlyCSEPass on test
