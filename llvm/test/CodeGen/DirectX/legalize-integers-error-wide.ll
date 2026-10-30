; RUN: not opt -passes=dxil-legalize -disable-output %s 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.3-library"

; CHECK: DXIL does not support integer types wider than 64 bits
define void @wide(i128 %v) {
  %r = add i128 %v, 1
  ret void
}
