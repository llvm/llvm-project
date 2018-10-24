; RUN: lli -jit-kind=orc-lazy -extra-module %p/Inputs/hidden-definitions.ll %s
; RUN: not lli -jit-kind=orc-lazy -jd libFoo -extra-module %p/Inputs/hidden-definitions.ll %s
;
; Check that hidden symbols in another module are visible when the module is
; added to the same JITDylib, and not visible if it is added to a different
; JITDylib.

@bar = external global i32
declare i32 @foo()

define i32 @main(i32 %argc, i8** nocapture readnone %argv) {
entry:
  %0 = call i32() @foo()
  %1 = load i32, i32* @bar
  %2 = add i32 %0, %1
  ret i32 %2
}
