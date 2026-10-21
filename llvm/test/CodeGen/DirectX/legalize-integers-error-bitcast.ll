; RUN: not opt -passes=dxil-legalize -disable-output %s 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.3-library"

; CHECK: DXIL legalization does not support this integer bitcast
define i32 @bitcast(<3 x i3> %v) {
  %p = bitcast <3 x i3> %v to i9
  %r = zext i9 %p to i32
  ret i32 %r
}
