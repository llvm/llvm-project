; RUN: llc -O0 -optimize-regalloc -regalloc=basic < %s
; This isn't exactly a useful set of command-line options, but check that it
; doesn't crash.  (It was crashing because a register was getting redefined.)

target triple = "thumbv7-apple-macosx10.6.7"

define i32 @f(i32* %x) nounwind ssp {
  %y = getelementptr inbounds i32* %x, i32 5000
  %tmp103 = load i32* %y, align 4
  ret i32 %tmp103
}
