; RUN: llc -load %llvm_obj_root/unittests/CodeGen/CGPluginTest/CGTestPlugin%pluginext %s -o - | FileCheck %s
; REQUIRES: native, system-linux, llvm-dylib

; CHECK: CodeGen Test Pass running on main
define void @main() {
  ret void
}
