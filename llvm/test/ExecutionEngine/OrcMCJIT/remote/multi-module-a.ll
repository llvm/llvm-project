; RUN: %lli -jit-kind=orc-mcjit -extra-module=%p/Inputs/multi-module-b.ll -extra-module=%p/Inputs/multi-module-c.ll -disable-lazy-compilation=true -remote-mcjit -mcjit-remote-process=lli-child-target%exeext %s > /dev/null
; XFAIL: win32

declare i32 @FB()

define i32 @main() nounwind {
  %r = call i32 @FB( )   ; <i32> [#uses=1]
  ret i32 %r
}

