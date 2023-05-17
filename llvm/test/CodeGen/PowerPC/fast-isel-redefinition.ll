; RUN: llc -O0 -verify-machineinstrs -fast-isel-abort=1 -optimize-regalloc -regalloc=basic -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s
; This isn't exactly a useful set of command-line options, but check that it
; doesn't crash.  (It crashed formerly on ARM, and proved useful in
; discovering a bug on PowerPC as well.)

define i32 @f(ptr %x) nounwind {
  %y = getelementptr inbounds i32, ptr %x, i32 5000
  %tmp103 = load i32, ptr %y, align 4
  ret i32 %tmp103
}
