; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: invalid vector element type

define void @bad() {
  %v = alloca <2 x target("spirv.Image")>
  ret void
}