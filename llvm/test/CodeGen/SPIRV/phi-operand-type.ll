; Check that requirements depending on an operand's type are derived correctly
; when that operand is a PHI, which has no result type operand until it is
; lowered to OpPhi.

; RUN: split-file %s %t
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product %t/main.ll -o - | FileCheck %t/main.ll
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product %t/main.ll -o - -filetype=obj | spirv-val %}
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %t/bf16.ll -o /dev/null 2>&1 | FileCheck %t/bf16.ll --check-prefix=CHECK-ERROR
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %t/bf16.ll -o - | FileCheck %t/bf16.ll

;--- main.ll
; CHECK-DAG: OpCapability Int64Atomics
; CHECK-DAG: OpCapability DotProductInput4x8Bit
; CHECK: OpFOrdLessThan
; CHECK: OpAtomicStore
; CHECK: OpSDot

define spir_func i1 @fcmp(float %x, float %y, i1 %c) {
entry:
  br i1 %c, label %then, label %exit

then:
  br label %exit

exit:
  %v = phi float [ %x, %entry ], [ %y, %then ]
  %r = fcmp olt float %v, %y
  ret i1 %r
}

define spir_func void @atomic_store(ptr addrspace(1) %p, i64 %x, i64 %y, i1 %c) {
entry:
  br i1 %c, label %then, label %exit

then:
  br label %exit

exit:
  %v = phi i64 [ %x, %entry ], [ %y, %then ]
  store atomic i64 %v, ptr addrspace(1) %p seq_cst, align 8
  ret void
}

define spir_func i32 @dot(<4 x i8> %x, <4 x i8> %y, i1 %c) {
entry:
  br i1 %c, label %then, label %exit

then:
  br label %exit

exit:
  %v = phi <4 x i8> [ %x, %entry ], [ %y, %then ]
  %r = call spir_func i32 @_Z3dotDv4_cS_(<4 x i8> %v, <4 x i8> %y)
  ret i32 %r
}

declare spir_func i32 @_Z3dotDv4_cS_(<4 x i8>, <4 x i8>)

;--- bf16.ll
; CHECK-ERROR: LLVM ERROR: Relational instructions with bfloat16 arguments require the following SPIR-V extension: SPV_INTEL_bfloat16_arithmetic

; CHECK-DAG: OpCapability BFloat16ArithmeticINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_bfloat16_arithmetic"
; CHECK: OpFOrdLessThan

define spir_func i1 @fcmp_bf16(bfloat %x, bfloat %y, i1 %c) {
entry:
  br i1 %c, label %then, label %exit

then:
  br label %exit

exit:
  %v = phi bfloat [ %x, %entry ], [ %y, %then ]
  %r = fcmp olt bfloat %v, %y
  ret i1 %r
}
