; RUN: llc -load %llvmshlibdir/CGTestPlugin%pluginext %s -o - | FileCheck %s
; REQUIRES: native, system-linux, llvm-dylib

; CHECK: CodeGen Test Pass running on main
define void @main() {
  ret void
}
