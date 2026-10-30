; RUN: not opt -passes=dxil-legalize -disable-output %s 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.3-library"

; CHECK: DXIL legalization does not support integer operand for instruction 'call'
declare void @takes_i3(i3)
define void @call() {
  call void @takes_i3(i3 1)
  ret void
}
