; RUN: not lli -jit-kind=orc-lazy -extra-object %p/Inputs/empty-object-file.o %s 2>&1 | FileCheck %s
;
; Test that bad object files yield an error.

; CHECK: The file was not recognized as a valid object file
define i32 @main(i32 %argc, i8** %argv) {
entry:
  ret i32 0
}
