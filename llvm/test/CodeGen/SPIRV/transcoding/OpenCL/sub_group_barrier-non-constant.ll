; RUN: split-file %s %t
; RUN: not --crash llc -O0 -mtriple=spirv32-unknown-unknown %t/flags.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FLAGS
; RUN: not --crash llc -O0 -mtriple=spirv64-unknown-unknown %t/flags.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FLAGS
; RUN: not --crash llc -O0 -mtriple=spirv32-unknown-unknown %t/scope.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SCOPE
; RUN: not --crash llc -O0 -mtriple=spirv64-unknown-unknown %t/scope.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SCOPE

; OpenCL allows runtime flags and scopes, but the backend currently only
; supports constants. Diagnose this limitation before GlobalISel aborts on
; the failed call lowering.
; FLAGS: error: {{.*}}in function runtime_flags void (i32): sub_group_barrier with non-constant arguments is not supported
; FLAGS: LLVM ERROR: unable to translate instruction: call (in function: runtime_flags)
; SCOPE: error: {{.*}}in function runtime_scope void (i32): sub_group_barrier with non-constant arguments is not supported
; SCOPE: LLVM ERROR: unable to translate instruction: call (in function: runtime_scope)

;--- flags.ll
define spir_kernel void @runtime_flags(i32 %flags) {
  call spir_func void @_Z17sub_group_barrierj(i32 %flags)
  ret void
}

declare spir_func void @_Z17sub_group_barrierj(i32)

;--- scope.ll
define spir_kernel void @runtime_scope(i32 %scope) {
  call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 2, i32 %scope)
  ret void
}

declare spir_func void @_Z17sub_group_barrierj12memory_scope(i32, i32)
