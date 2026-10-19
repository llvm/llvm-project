; RUN: split-file %s %t
; RUN: not opt -passes=dxil-legalize -disable-output %t/wide.ll 2>&1 | FileCheck %s --check-prefix=WIDE
; RUN: not opt -passes=dxil-legalize -disable-output %t/phi.ll 2>&1 | FileCheck %s --check-prefix=PHI
; RUN: not opt -passes=dxil-legalize -disable-output %t/argument.ll 2>&1 | FileCheck %s --check-prefix=ARG
; RUN: not opt -passes=dxil-legalize -disable-output %t/return.ll 2>&1 | FileCheck %s --check-prefix=RET
; RUN: not opt -passes=dxil-legalize -disable-output %t/bitcast.ll 2>&1 | FileCheck %s --check-prefix=BITCAST
; RUN: not opt -passes=dxil-legalize -disable-output %t/call.ll 2>&1 | FileCheck %s --check-prefix=CALL

;--- wide.ll
target triple = "dxil-pc-shadermodel6.3-library"
; WIDE: DXIL does not support integer types wider than 64 bits
define void @wide(i128 %v) {
  %r = add i128 %v, 1
  ret void
}

;--- phi.ll
target triple = "dxil-pc-shadermodel6.3-library"
; PHI: DXIL legalization does not support integer result for instruction 'phi'
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

;--- argument.ll
target triple = "dxil-pc-shadermodel6.3-library"
; ARG: DXIL legalization is missing an integer operand replacement
define i32 @argument(i3 %v) {
  %r = zext i3 %v to i32
  ret i32 %r
}

;--- return.ll
target triple = "dxil-pc-shadermodel6.3-library"
; RET: DXIL legalization does not support integer operand for instruction 'ret'
define i3 @narrow_return() {
  ret i3 0
}

;--- bitcast.ll
target triple = "dxil-pc-shadermodel6.3-library"
; BITCAST: DXIL legalization does not support this integer bitcast
define i32 @bitcast(<3 x i3> %v) {
  %p = bitcast <3 x i3> %v to i9
  %r = zext i9 %p to i32
  ret i32 %r
}

;--- call.ll
target triple = "dxil-pc-shadermodel6.3-library"
; CALL: DXIL legalization does not support integer operand for instruction 'call'
declare void @takes_i3(i3)
define void @call() {
  call void @takes_i3(i3 1)
  ret void
}
