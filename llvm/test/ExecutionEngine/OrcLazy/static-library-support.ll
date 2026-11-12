; This first line will generate the .o files for the next run line
; RUN: rm -rf %t && mkdir -p %t
; RUN: llc -filetype=obj -o %t/foo.o %p/Inputs/foo-return-i32-0.ll
; RUN: llc -filetype=obj -o %t/bar.o %p/Inputs/bar-return-i32-call-foo.ll
; RUN: llvm-ar r %t/staticlib.a %t/foo.o %t/bar.o
; RUN: lli -jit-kind=orc-lazy -extra-archive %t/staticlib.a %s

declare i32 @bar()

define i32 @main() {
  %r = call i32 @bar()   ; <i32> [#uses=1]
  ret i32 %r
}
