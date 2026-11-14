; RUN: split-file %s %t
; RUN: llc -O0 -mtriple=spirv64v1.1-unknown-unknown %t/max-byte-offset.ll -o - | FileCheck %t/max-byte-offset.ll
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.1-unknown-unknown %t/max-byte-offset.ll -o - -filetype=obj | spirv-val --target-env spv1.1 %}
; RUN: not --crash llc -O0 -mtriple=spirv64v1.0-unknown-unknown %t/max-byte-offset.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
; RUN: not --crash llc -O0 -mtriple=spirv64v1.1-unknown-unknown %t/alignment-id.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
; RUN: not --crash llc -O0 -mtriple=spirv64v1.1-unknown-unknown %t/max-byte-offset-id.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
; RUN: llc -O0 -mtriple=spirv32v1.4-unknown-vulkan %t/uniform-id.ll -o - | FileCheck %t/uniform-id.ll
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.4-unknown-vulkan %t/uniform-id.ll -o - -filetype=obj | spirv-val --target-env spv1.4 %}
; RUN: not --crash llc -O0 -mtriple=spirv32v1.3-unknown-vulkan %t/uniform-id.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
; RUN: llc -O0 -mtriple=spirv32v1.3-unknown-vulkan %t/buffer-block.ll -o - | FileCheck %t/buffer-block.ll
; RUN: not --crash llc -O0 -mtriple=spirv32v1.4-unknown-vulkan %t/buffer-block.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERR

; MaxByteOffset requires SPIR-V 1.1, AlignmentId and MaxByteOffsetId require
; 1.2, UniformId requires 1.4, BufferBlock is removed after 1.3.

; CHECK-ERR: LLVM ERROR: Adding SPIR-V requirements this target can't satisfy.

;--- max-byte-offset.ll
; CHECK: OpDecorate %[[#VAR:]] MaxByteOffset 12
; CHECK: %[[#VAR]] = OpVariable %[[#]] Function

define spir_func void @f() {
  %res = alloca i16, align 2, !spirv.Decorations !1
  ret void
}

!1 = !{!2}
!2 = !{i32 45, i32 12}

;--- alignment-id.ll
define spir_func void @f() {
  %res = alloca i16, align 2, !spirv.Decorations !1
  ret void
}

!1 = !{!2}
!2 = !{i32 46, i32 12}

;--- max-byte-offset-id.ll
define spir_func void @f() {
  %res = alloca i16, align 2, !spirv.Decorations !1
  ret void
}

!1 = !{!2}
!2 = !{i32 47, i32 12}

;--- uniform-id.ll
; CHECK: OpDecorateId %[[#]] UniformId %[[#]]

@uvar = external local_unnamed_addr addrspace(7) externally_initialized constant i32, !spirv.Decorations !1

define i32 @f() {
  %v = load i32, ptr addrspace(7) @uvar
  ret i32 %v
}

!1 = !{!2}
!2 = !{i32 27, i32 2}

;--- buffer-block.ll
; CHECK: OpDecorate %[[#]] BufferBlock

@uvar = external local_unnamed_addr addrspace(7) externally_initialized constant i32, !spirv.Decorations !1

define i32 @f() {
  %v = load i32, ptr addrspace(7) @uvar
  ret i32 %v
}

!1 = !{!2}
!2 = !{i32 3}
