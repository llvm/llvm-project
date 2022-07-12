; RUN: split-file %s %t
; RUN: not llvm-as < %t/parse-fail.ll 2>&1 | FileCheck %s --check-prefix=CHECK-PARSE-FAIL
; RUN: not llvm-as < %t/input-before-output.ll 2>&1 | FileCheck %s --check-prefix=CHECK-INPUT-BEFORE-OUTPUT
; RUN: not llvm-as < %t/input-after-clobber.ll 2>&1 | FileCheck %s --check-prefix=CHECK-INPUT-AFTER-CLOBBER
; RUN: not llvm-as < %t/must-return-void.ll 2>&1 | FileCheck %s --check-prefix=CHECK-MUST-RETURN-VOID
; RUN: not llvm-as < %t/cannot-be-struct.ll 2>&1 | FileCheck %s --check-prefix=CHECK-CANNOT-BE-STRUCT
; RUN: not llvm-as < %t/incorrect-struct-elements.ll 2>&1 | FileCheck %s --check-prefix=CHECK-INCORRECT-STRUCT-ELEMENTS
; RUN: not llvm-as < %t/incorrect-arg-num.ll 2>&1 | FileCheck %s --check-prefix=CHECK-INCORRECT-ARG-NUM

;--- parse-fail.ll
; CHECK-PARSE-FAIL: failed to parse constraints
define void @foo() {
  ; "~x{21}" is not a valid clobber constraint.
  call void asm sideeffect "mov x0, #42", "~{x0},~{x19},~x{21}"()
  ret void
}

;--- input-before-output.ll
; CHECK-INPUT-BEFORE-OUTPUT: output constraint occurs after input or clobber constraint
define void @foo() {
  call void asm sideeffect "mov x0, #42", "r,=r"()
  ret void
}

;--- input-after-clobber.ll
; CHECK-INPUT-AFTER-CLOBBER: input constraint occurs after clobber constraint
define void @foo() {
  call void asm sideeffect "mov x0, #42", "~{x0},r"()
  ret void
}

;--- must-return-void.ll
; CHECK-MUST-RETURN-VOID: inline asm without outputs must return void
define void @foo() {
  call i32 asm sideeffect "mov x0, #42", ""()
  ret void
}

;--- cannot-be-struct.ll
; CHECK-CANNOT-BE-STRUCT: inline asm with one output cannot return struct
define void @foo() {
  call { i32 } asm sideeffect "mov x0, #42", "=r"()
  ret void
}

;--- incorrect-struct-elements.ll
; CHECK-INCORRECT-STRUCT-ELEMENTS: number of output constraints does not match number of return struct elements
define void @foo() {
  call { i32 } asm sideeffect "mov x0, #42", "=r,=r"()
  ret void
}

;--- incorrect-arg-num.ll
; CHECK-INCORRECT-ARG-NUM: number of input constraints does not match number of parameters
define void @foo() {
  call void asm sideeffect "mov x0, #42", "r"()
  ret void
}
