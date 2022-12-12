; RUN: %lli -jit-kind=mcjit -extra-module=%p/Inputs/cross-module-b.ll -disable-lazy-compilation=true -remote-mcjit -mcjit-remote-process=lli-child-target%exeext %s > /dev/null
; XFAIL: target={{.*-windows-(gnu|msvc)}}
; UNSUPPORTED: powerpc64-unknown-linux-gnu
; Remove UNSUPPORTED for powerpc64-unknown-linux-gnu if problem caused by r266663 is fixed

declare i32 @FB()

define i32 @FA() nounwind {
  ret i32 0
}

define i32 @main() nounwind {
  %r = call i32 @FB( )   ; <i32> [#uses=1]
  ret i32 %r
}
