; RUN: rm -rf %t && split-file %s %t

; RUN: not llvm-as < %t/not_a_constant.ll -o /dev/null 2>&1 | FileCheck -check-prefix=NOT_A_CONSTANT %s
; RUN: not llvm-as < %t/not_a_sclar.ll -o /dev/null 2>&1 | FileCheck -check-prefix=NOT_A_SCALAR %s
; RUN: not llvm-as < %t/not_a_vector.ll -o /dev/null 2>&1 | FileCheck -check-prefix=NOT_A_VECTOR %s
; RUN: not llvm-as < %t/wrong_explicit_type.ll -o /dev/null 2>&1 | FileCheck -check-prefix=WRONG_EXPLICIT_TYPE %s
; RUN: not llvm-as < %t/wrong_implicit_type.ll -o /dev/null 2>&1 | FileCheck -check-prefix=WRONG_IMPLICIT_TYPE %s

;--- not_a_constant.ll
; NOT_A_CONSTANT: error: expected instruction opcode
define <4 x i32> @not_a_constant(i32 %a) {
  %splat = splat (i32 %a)
  ret <vscale x 4 x i32> %splat
}

;--- not_a_sclar.ll
; NOT_A_SCALAR: error: constant expression type mismatch: got type '<1 x i32>' but expected 'i32'
define <4 x i32> @not_a_scalar() {
  ret <4 x i32> splat (<1 x i32> <i32 7>)
}

;--- not_a_vector.ll
; NOT_A_VECTOR: error: vector constant must have vector type
define <4 x i32> @not_a_vector() {
  ret i32 splat (i32 7)
}

;--- wrong_explicit_type.ll
; WRONG_EXPLICIT_TYPE: error: constant expression type mismatch: got type 'i8' but expected 'i32'
define <4 x i32> @wrong_explicit_type() {
  ret <4 x i32> splat (i8 7)
}

;--- wrong_implicit_type.ll
; WRONG_IMPLICIT_TYPE: error: constant expression type mismatch: got type 'i8' but expected 'i32'
define void @wrong_implicit_type(<4 x i32> %a) {
  %add = add <4 x i32> %a, splat (i8 7)
  ret void
}

