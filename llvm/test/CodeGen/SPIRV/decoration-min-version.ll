; RUN: llc -O0 -mtriple=spirv64v1.1-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.1-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.1 %}
; RUN: not --crash llc -O0 -mtriple=spirv64v1.0-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERR

; MaxByteOffset requires SPIR-V 1.1.

; CHECK: OpDecorate %[[#VAR:]] MaxByteOffset 12
; CHECK: %[[#VAR]] = OpVariable %[[#]] Function

; CHECK-ERR: LLVM ERROR: Adding SPIR-V requirements this target can't satisfy.

define spir_func void @f() {
  %res = alloca i16, align 2, !spirv.Decorations !1
  ret void
}

!1 = !{!2}
!2 = !{i32 45, i32 12}
