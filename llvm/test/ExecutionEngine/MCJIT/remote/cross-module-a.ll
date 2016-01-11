; RUN: %lli -extra-module=%p/Inputs/cross-module-b.ll -disable-lazy-compilation=true -remote-mcjit -mcjit-remote-process=lli-child-target%exeext %s > /dev/null
; XFAIL: win32

declare i32 @FB()

define i32 @FA() nounwind {
  ret i32 0
}

define i32 @main() nounwind {
  %r = call i32 @FB( )   ; <i32> [#uses=1]
  ret i32 %r
}
