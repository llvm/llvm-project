; REQUIRES: system-windows
; RUN: lli -jit-kind=orc-lazy -extra-module %p/Inputs/comdat-functions.ll %s
; Check if crashing comdat any functions are not causing duplicate symbol error.

$baz = comdat any

define i32 @baz() comdat {
entry:
  ret i32 0
}

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %call = tail call i32 @baz()
  ret i32 %call
}
