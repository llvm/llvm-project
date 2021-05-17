; RUN: %lli -jit-kind=mcjit -extra-module=%p/Inputs/multi-module-b.ll -extra-module=%p/Inputs/multi-module-c.ll -relocation-model=pic -code-model=small %s > /dev/null
; RUN: %lli -lljit-platform=Inactive -extra-module=%p/Inputs/multi-module-b.ll -extra-module=%p/Inputs/multi-module-c.ll -relocation-model=pic -code-model=small %s > /dev/null
; XFAIL: mips-, mipsel-, i686, i386

declare i32 @FB()

define i32 @main() {
  %r = call i32 @FB( )   ; <i32> [#uses=1]
  ret i32 %r
}

