; RUN: split-file %s %t
; RUN: not llvm-as < %t/extractvalue.ll 2>&1 | FileCheck %s --check-prefix=EXTRACTVALUE
; RUN: not llvm-as < %t/insertvalue.ll 2>&1 | FileCheck %s --check-prefix=INSERTVALUE

;--- extractvalue.ll
define i32 @extractvalue() {
; EXTRACTVALUE: error: extractvalue constexprs are no longer supported
  ret i32 extractvalue ({i32} {i32 3}, 0)
}

;--- insertvalue.ll
define {i32} @insertvalue() {
; INSERTVALUE: error: insertvalue constexprs are no longer supported
  ret {i32} insertvalue ({i32} poison, i32 3, 0)
}
