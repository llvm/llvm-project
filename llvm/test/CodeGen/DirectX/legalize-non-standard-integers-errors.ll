; RUN: split-file %s %t
; RUN: not opt -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %t/wide.ll -disable-output 2>&1 | FileCheck %s --check-prefix=WIDE
; RUN: not opt -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %t/bitcast.ll -disable-output 2>&1 | FileCheck %s --check-prefix=BITCAST
; RUN: not opt -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %t/phi.ll -disable-output 2>&1 | FileCheck %s --check-prefix=PHI
; RUN: not opt -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %t/operand.ll -disable-output 2>&1 | FileCheck %s --check-prefix=OPERAND
; RUN: not opt -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %t/return.ll -disable-output 2>&1 | FileCheck %s --check-prefix=RETURN

;--- wide.ll

define void @integer_too_wide(i128 %lhs, i128 %rhs) {
; WIDE: LLVM ERROR: DXIL does not support integer types wider than 64 bits
  %sum = add i128 %lhs, %rhs
  ret void
}

;--- bitcast.ll

define <3 x i1> @unsupported_reverse_bitcast(i32 %value) {
; BITCAST: LLVM ERROR: DXIL legalization does not support this integer bitcast
  %narrow = trunc i32 %value to i3
  %result = bitcast i3 %narrow to <3 x i1>
  ret <3 x i1> %result
}

;--- phi.ll

define i32 @unsupported_phi(i1 %condition, i32 %value) {
; PHI: LLVM ERROR: DXIL legalization does not support non-standard integer result type for instruction 'phi'
entry:
  %narrow = trunc i32 %value to i3
  br i1 %condition, label %left, label %right

left:
  br label %exit

right:
  br label %exit

exit:
  %result = phi i3 [ %narrow, %left ], [ 0, %right ]
  %extended = zext i3 %result to i32
  ret i32 %extended
}

;--- operand.ll

define i32 @missing_operand_replacement(i3 %value) {
; OPERAND: LLVM ERROR: DXIL legalization is missing an integer operand replacement
  %sum = add i3 %value, 1
  %result = zext i3 %sum to i32
  ret i32 %result
}

;--- return.ll

define i3 @unsupported_return() {
; RETURN: LLVM ERROR: DXIL legalization does not support non-standard integer operand type for instruction 'ret'
  ret i3 0
}
