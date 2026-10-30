; RUN: not opt -passes=dxil-legalize -disable-output %s 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.3-library"

; CHECK: DXIL legalization does not support integer operand for instruction 'ret'
define i3 @narrow_return() {
  ret i3 0
}
