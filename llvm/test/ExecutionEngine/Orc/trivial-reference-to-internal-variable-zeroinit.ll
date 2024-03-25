; Check that we can execute a program that makes a single reference to an
; internal linkage global variable that is zero-initialized.
;
; Failure may indicate a problem with zero-initialized sections, or data
; sections more generally.
;
; RUN: %lli %s

@X = internal global i32 0

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %0 = load i32, i32* @X
  ret i32 %0
}
