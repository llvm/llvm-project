;RUN:  not llc -mtriple=aarch64-linux-gnu -mattr=-fp-armv8 < %s 2>&1 | FileCheck %s

; CHECK: error: couldn't allocate output register for constraint '{d0}'
; CHECK: error: couldn't allocate output register for constraint 'w'

define hidden double @test1(double %xx) local_unnamed_addr #0 {
entry:
  %0 = tail call double asm "frintp ${0:d}, ${0:d}", "={d0}"()
  ret double %0
}

define hidden double @test2(double %xx) local_unnamed_addr #0 {
entry:
  %0 = tail call double asm "frintp ${0:d}, ${0:d}", "=w"()
  ret double %0
}

