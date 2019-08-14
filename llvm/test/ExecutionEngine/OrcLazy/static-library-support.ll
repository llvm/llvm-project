; This first line will generate the .o files for the next run line
; RUN: llc -filetype=obj -o %t.o %p/Inputs/basic-object-source.ll
; RUN: llvm-ar r %t.a %t.o
; RUN: lli -jit-kind=orc-lazy -extra-archive %t.a %s

declare i32 @foo()

define i32 @main() {
  %r = call i32 @foo( )   ; <i32> [#uses=1]
  ret i32 %r
}
