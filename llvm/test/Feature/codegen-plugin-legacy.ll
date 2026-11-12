; RUN: llc -load-pass-plugin %llvmshlibdir/Ciao%pluginext %s -o - | FileCheck %s
; REQUIRES: native, system-linux, llvm-dylib, plugins, examples

; CHECK: Ciao: main
define void @main() {
  ret void
}
