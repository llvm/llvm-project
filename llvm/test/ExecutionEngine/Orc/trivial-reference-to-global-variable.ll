; Check that we can execute a program that makes a single reference to an
; external linkage global variable that is initialized to a non-zero value.
;
; Failure may indicate a problem with data-section or GOT handling.
;
; RUN: %lli %s

@X = global i32 1

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %0 = load i32, i32* @X
  %1 = icmp ne i32 %0, 1
  %2 = zext i1 %1 to i32
  ret i32 %2
}
