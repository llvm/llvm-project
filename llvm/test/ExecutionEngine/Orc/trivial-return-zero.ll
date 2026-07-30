; Check that we can execute a program that does nothing and just returns zero.
;
; This is the simplest possible JIT smoke test. If it fails it indicates a
; critical failure in the JIT (e.g. failure to set memory permissions) that's
; likely to affect all programs.
;
; RUN: %lli %s

define i32 @main(i32 %argc, i8** %argv)  {
  ret i32 0
}
