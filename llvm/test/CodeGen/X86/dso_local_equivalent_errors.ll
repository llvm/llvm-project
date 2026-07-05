; RUN: split-file %s %t
; RUN: not llc -mtriple=x86_64-linux-gnu -o - %t/undefined_func.ll 2>&1 | FileCheck %s -check-prefix=UNDEFINED
; RUN: not llc -mtriple=x86_64-linux-gnu -o - %t/invalid_arg.ll 2>&1 | FileCheck %s -check-prefix=INVALID

;--- undefined_func.ll
; UNDEFINED: error: unknown function 'undefined_func' referenced by dso_local_equivalent
define void @call_undefined() {
  call void dso_local_equivalent @undefined_func()
  ret void
}

;--- invalid_arg.ll
; INVALID: error: expected a function, alias to function, or ifunc in dso_local_equivalent
define void @call_global_var() {
  call void dso_local_equivalent @glob()
  ret void
}

@glob = constant i32 1
