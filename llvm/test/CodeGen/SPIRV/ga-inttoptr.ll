; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown < %s 2>&1 | FileCheck %s
; CHECK: argument of incompatible type!

; The BE does not support non-global-object aliases
@glob.inttoptr = alias i32, inttoptr (i64 12345 to ptr)

define spir_kernel void @kernel() addrspace(4) {
entry:
  store i32 0, ptr @glob.inttoptr
  ret void
}
