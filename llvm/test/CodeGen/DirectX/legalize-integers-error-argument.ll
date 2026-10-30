; RUN: not opt -passes=dxil-legalize -disable-output %s 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.3-library"

; CHECK: DXIL legalization is missing an integer operand replacement
define i32 @argument(i3 %v) {
  %r = zext i3 %v to i32
  ret i32 %r
}
