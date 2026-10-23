; RUN: not opt -passes=dxil-legalize -disable-output %s 2>&1 | FileCheck %s

target triple = "dxil-pc-shadermodel6.3-library"

; CHECK: DXIL legalization does not support integer result for instruction 'phi'
define i32 @phi(i1 %c) {
entry:
  br i1 %c, label %left, label %end
left:
  br label %end
end:
  %p = phi i3 [ 0, %entry ], [ 1, %left ]
  %r = zext i3 %p to i32
  ret i32 %r
}
