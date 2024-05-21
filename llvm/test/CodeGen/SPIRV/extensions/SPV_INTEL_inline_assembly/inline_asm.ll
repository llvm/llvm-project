; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_INTEL_inline_assembly -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_INTEL_inline_assembly -o - -filetype=obj | spirv-val %}

; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: LLVM ERROR: requires the following SPIR-V extension: SPV_INTEL_inline_assembly

; CHECK: Capability AsmINTEL
; CHECK: Extension "SPV_INTEL_inline_assembly"
; CHECK: Decorate SideEffectsINTEL
; CHECK: AsmTargetINTEL "spir64-unknown-unknown"
; CHECK: AsmINTEL
; CHECK: OpFunction
; CHECK: AsmCallINTEL

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define spir_kernel void @foo(ptr addrspace(1) %_arg_int, ptr addrspace(1) %_arg_float, ptr addrspace(1) %_arg_half, i64 %_lng) {
  %i1 = load i32, ptr addrspace(1) %_arg_int
  %i2 = load i8, ptr addrspace(1) %_arg_int
  %f1 = load float, ptr addrspace(1) %_arg_float
  %h1 = load half, ptr addrspace(1) %_arg_half
;  ; inline asm: complex result
;  call { i64, half } asm sideeffect "structcmd_nop", "=r,=r"()
;  %r_struct = call { i64, half } asm sideeffect "structcmd $0 $0 $1 $1", "=r,=r,0,1"(i64 123, half %h1)
;;  %r_struct1 = extractvalue { i64, half } %r_struct, 0
;;  %r_struct2 = extractvalue { i64, half } %r_struct, 1
;;  store half %r_struct2, ptr addrspace(1) %_arg_half
  ; inline asm
  call void asm sideeffect "", ""()
  call void asm sideeffect "undefined\0A", ""()
  call void asm sideeffect "", "~{cc},~{memory}"()
  %res_i0 = call i32 asm "clobber_out $0", "=&r"()
  store i32 %res_i0, ptr addrspace(1) %_arg_int
  ; inline asm: integer
  %res_i1 = call i32 asm sideeffect "icmd $0 $1", "=r,r"(i32 %i1)
  store i32 %res_i1, ptr addrspace(1) %_arg_int
  ; inline asm: float
  %res_f1 = call float asm sideeffect "fcmd $0 $1", "=r,r"(float %f1)
  store float %res_f1, ptr addrspace(1) %_arg_float
  ; inline asm: mixed floats
  %res_f2 = call half asm sideeffect "fcmdext $0 $1 $2", "=r,r,r"(float 2.0, float %f1)
  store float %res_f1, ptr addrspace(1) %_arg_half
  ; inline asm: mixed operands of different types
  call i8 asm sideeffect "cmdext $0 $3 $1 $2", "=r,r,r,r"(float %f1, i32 123, i8 %i2)
  ; inline asm: mixed integers
  %res_i2 = call i64 asm sideeffect "icmdext $0 $3 $1 $2", "=r,r,r,r"(i64 %_lng, i32 %i1, i8 %i2)
  store i64 %res_i2, ptr addrspace(1) %_arg_int
  ; inline asm: constant arguments
  call void asm sideeffect "constcmd $0 $1", "i,i"(i32 123, double 42.0)
  ret void
}
