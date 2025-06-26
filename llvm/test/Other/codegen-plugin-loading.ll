; RUN: llc -load %llvmshlibdir/../unittests/CodeGen/CGPluginTest/CGTestPlugin%pluginext %s -o - | FileCheck %s

; CHECK: CodeGen Test Pass running on main
define void @main() {
  ret void
}
