; This first line will generate the .o files for the next run line
; RUN: rm -rf %t.cachedir %t.cachedir2 %t.cachedir3
; RUN: mkdir -p %t.cachedir %t.cachedir2 %t.cachedir3
; RUN: %lli -jit-kind=mcjit -extra-module=%p/Inputs/multi-module-b.ll -extra-module=%p/Inputs/multi-module-c.ll -enable-cache-manager -object-cache-dir=%t.cachedir %s
; RUN: %lli -extra-module=%p/Inputs/multi-module-b.ll -extra-module=%p/Inputs/multi-module-c.ll -enable-cache-manager -object-cache-dir=%t.cachedir %s

; Collect generated objects.
; RUN: find %t.cachedir -type f -name 'multi-module-?.o' -exec mv -v '{}' %t.cachedir2 ';'

; This line tests MCJIT object loading
; RUN: %lli -jit-kind=mcjit -extra-object=%t.cachedir2/multi-module-b.o -extra-object=%t.cachedir2/multi-module-c.o %s
; RUN: %lli -extra-object=%t.cachedir2/multi-module-b.o -extra-object=%t.cachedir2/multi-module-c.o %s

; These lines put the object files into an archive
; RUN: llvm-ar r %t.cachedir3/load-object.a %t.cachedir2/multi-module-b.o
; RUN: llvm-ar r %t.cachedir3/load-object.a %t.cachedir2/multi-module-c.o

; This line test MCJIT archive loading
; RUN: %lli -jit-kind=mcjit -extra-archive=%t.cachedir3/load-object.a %s
; RUN: %lli -extra-archive=%t.cachedir3/load-object.a %s

declare i32 @FB()

define i32 @main() {
  %r = call i32 @FB( )   ; <i32> [#uses=1]
  ret i32 %r
}
