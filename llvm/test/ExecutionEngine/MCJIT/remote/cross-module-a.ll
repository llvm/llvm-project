; RUN: %lli_mcjit -extra-module=%p/Inputs/cross-module-b.ll -disable-lazy-compilation=true -remote-mcjit -mcjit-remote-process=lli-child-target %s > /dev/null

declare i32 @FB()

define i32 @FA() {
  ret i32 0
}

define i32 @main() {
  %r = call i32 @FB( )   ; <i32> [#uses=1]
  ret i32 %r
}

