; Check that we can forward PreCodeGenCallback to llvm::runCodeGenPipeline()
; REQUIRES: system-linux, plugins, examples, x86-registered-target

; Resolve runCodeGenPipeline() from libLLVM, since llc doesn't use it yet
; REQUIRES: llvm-dylib

; The plugin prefixes functions with ciao_ and forwards to runCodeGenPipeline()
; RUN: llc %s -o %t.o -mtriple=x86_64 -filetype=obj \
; RUN:        -load-pass-plugin %llvmshlibdir/Ciao%pluginext \
; RUN:        -plugin-arg=Ciao,forward-codegen
; RUN: llvm-nm -j %t.o | FileCheck %s

; CHECK: {{^}}ciao_main{{$}}

define void @main() {
  ret void
}
