; Check that we can execute a program that makes a single call to an external
; linkage function that returns zero.
;
; Failure may indicate a problem with branch, GOT, or PLT relocation handling
; in the JIT linker.
;
; RUN: %lli %s

define i32 @foo() {
  ret i32 0
}

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %0 = call i32 @foo()
  ret i32 %0
}
