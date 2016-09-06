; RUN: %lli -jit-kind=orc-mcjit -extra-module=%p/Inputs/multi-module-b.ll -extra-module=%p/Inputs/multi-module-c.ll -disable-lazy-compilation=true -remote-mcjit -mcjit-remote-process=lli-child-target%exeext %s > /dev/null
; XFAIL: mingw32,win32
; UNSUPPORTED: powerpc64-unknown-linux-gnu
; Remove UNSUPPORTED for powerpc64-unknown-linux-gnu if problem caused by r266663 is fixed

declare i32 @FB()

define i32 @main() nounwind {
  %r = call i32 @FB( )   ; <i32> [#uses=1]
  ret i32 %r
}

