; RUN: llc -filetype=obj -o %t %p/Inputs/foo-return-i32-0.ll
; RUN: lli -jit-kind=orc-lazy -extra-object %t %s
;
; Check that we can load an object file and call a function in it.

declare i32 @foo()

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %0 = call i32 @foo()
  ret i32 %0
}

