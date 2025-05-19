; RUN: llvm-as -o - %s | llvm-dis | FileCheck %s

; CHECK-LABEL: @vec_ops(
define <2 x target("spirv.Image")> @vec_ops(<2 x target("spirv.Image")> %x) {
  %a = alloca <2 x target("spirv.Image")>
  store <2 x target("spirv.Image")> %x, ptr %a
  %load = load <2 x target("spirv.Image")>, ptr %a
  %elt = extractelement <2 x target("spirv.Image")> %load, i64 0
  %res = insertelement <2 x target("spirv.Image")> undef, target("spirv.Image") %elt, i64 1
  ret <2 x target("spirv.Image")> %res
}