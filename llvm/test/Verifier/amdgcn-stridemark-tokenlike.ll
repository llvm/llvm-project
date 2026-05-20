; RUN: split-file %s %t
; RUN: not llvm-as %t/function-arg.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-ARG %s
; RUN: not llvm-as %t/function-return.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-RETURN %s

;--- function-arg.ll
define void @f(target("amdgcn.stridemark") %mark) {
  ret void
}
; CHECK-ARG: Function takes token but isn't an intrinsic

;--- function-return.ll
define target("amdgcn.stridemark") @f() {
  ret target("amdgcn.stridemark") poison
}
; CHECK-RETURN: Function returns a token but isn't an intrinsic
