; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_INTEL_inline_assembly -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_INTEL_inline_assembly -o - -filetype=obj | spirv-val %}

; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: Inline assembly instructions require the following SPIR-V extension: SPV_INTEL_inline_assembly

; CHECK: OpCapability AsmINTEL
; CHECK: OpExtension "SPV_INTEL_inline_assembly"

; CHECK-COUNT-8: OpDecorate %[[#]] SideEffectsINTEL

; CHECK-DAG: %[[#VoidTy:]] = OpTypeVoid
; CHECK-DAG: %[[#Int8Ty:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Int64Ty:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#HalfTy:]] = OpTypeFloat 16
; CHECK-DAG: %[[#FloatTy:]] = OpTypeFloat 32
; CHECK-DAG: %[[#DoubleTy:]] = OpTypeFloat 64

; CHECK-DAG: OpTypeFunction %[[#VoidTy]] %[[#]] %[[#]] %[[#]] %[[#Int64Ty]]
; CHECK-DAG: %[[#Fun1Ty:]] = OpTypeFunction %[[#VoidTy]]
; CHECK-DAG: %[[#Fun2Ty:]] = OpTypeFunction %[[#Int32Ty]]
; CHECK-DAG: %[[#Fun3Ty:]] = OpTypeFunction %[[#Int32Ty]] %[[#Int32Ty]]
; CHECK-DAG: %[[#Fun4Ty:]] = OpTypeFunction %[[#FloatTy]] %[[#FloatTy]]
; CHECK-DAG: %[[#Fun5Ty:]] = OpTypeFunction %[[#HalfTy]] %[[#FloatTy]] %[[#FloatTy]]
; CHECK-DAG: %[[#Fun6Ty:]] = OpTypeFunction %[[#Int8Ty]] %[[#FloatTy]] %[[#Int32Ty]] %[[#Int8Ty]]
; CHECK-DAG: %[[#Fun7Ty:]] = OpTypeFunction %[[#Int64Ty]] %[[#Int64Ty]] %[[#Int32Ty]] %[[#Int8Ty]]
; CHECK-DAG: %[[#Fun8Ty:]] = OpTypeFunction %[[#VoidTy]] %[[#Int32Ty]] %[[#DoubleTy]]

; CHECK-DAG: %[[#Const2:]] = OpConstant %[[#FloatTy]] 2
; CHECK-DAG: %[[#Const123:]] = OpConstant %[[#Int32Ty]] 123
; CHECK-DAG: %[[#Const42:]] = OpConstant %[[#DoubleTy:]] 42

; CHECK: %[[#Dialect:]] = OpAsmTargetINTEL "spirv64-unknown-unknown"
; CHECK-NO: OpAsmTargetINTEL

; CHECK: %[[#Asm1:]] = OpAsmINTEL %[[#VoidTy]] %[[#Fun1Ty]] %[[#Dialect]] "" ""
; CHECK: %[[#Asm2:]] = OpAsmINTEL %[[#VoidTy]] %[[#Fun1Ty]] %[[#Dialect]] "nop" ""
; CHECK: %[[#Asm3:]] = OpAsmINTEL %[[#VoidTy]] %[[#Fun1Ty]] %[[#Dialect]] "" "~{cc},~{memory}"
; CHECK: %[[#Asm4:]] = OpAsmINTEL %[[#Int32Ty]] %[[#Fun2Ty:]] %[[#Dialect]] "clobber_out $0" "=&r"
; CHECK: %[[#Asm5:]] = OpAsmINTEL %[[#Int32Ty]] %[[#Fun3Ty]] %[[#Dialect]] "icmd $0 $1" "=r,r"
; CHECK: %[[#Asm6:]] = OpAsmINTEL %[[#FloatTy]] %[[#Fun4Ty]] %[[#Dialect]] "fcmd $0 $1" "=r,r"
; CHECK: %[[#Asm7:]] = OpAsmINTEL %[[#HalfTy]] %[[#Fun5Ty]] %[[#Dialect]] "fcmdext $0 $1 $2" "=r,r,r"
; CHECK: %[[#Asm8:]] = OpAsmINTEL %[[#Int8Ty]] %[[#Fun6Ty]] %[[#Dialect]] "cmdext $0 $3 $1 $2" "=r,r,r,r"
; CHECK: %[[#Asm9:]] = OpAsmINTEL %[[#Int64Ty]] %[[#Fun7Ty]] %[[#Dialect]] "icmdext $0 $3 $1 $2" "=r,r,r,r"
; CHECK: %[[#Asm10:]] = OpAsmINTEL %[[#VoidTy]] %[[#Fun8Ty]] %[[#Dialect]] "constcmd $0 $1" "r,r"
; CHECK: %[[#Asm11:]] = OpAsmINTEL %[[#VoidTy]] %[[#Fun8Ty]] %[[#Dialect]] "constcmd $0 $1" "i,i"
; CHECK-NO: OpAsmINTEL

; CHECK: OpFunction
; CHECK: OpAsmCallINTEL %[[#VoidTy]] %[[#Asm1]]
; CHECK: OpAsmCallINTEL %[[#VoidTy]] %[[#Asm2]]
; CHECK: OpAsmCallINTEL %[[#VoidTy]] %[[#Asm3]]
; CHECK: OpAsmCallINTEL %[[#Int32Ty]] %[[#Asm4]]
; CHECK: OpAsmCallINTEL %[[#Int32Ty]] %[[#Asm5]] %[[#]]
; CHECK: OpAsmCallINTEL %[[#FloatTy]] %[[#Asm6]] %[[#]]
; CHECK: OpAsmCallINTEL %[[#HalfTy]] %[[#Asm7]] %[[#Const2]] %[[#]]
; CHECK: OpAsmCallINTEL %[[#Int8Ty]] %[[#Asm8]] %[[#]] %[[#Const123]] %[[#]]
; CHECK: OpAsmCallINTEL %[[#Int64Ty]] %[[#Asm9]] %[[#]] %[[#]] %[[#]]
; CHECK: OpAsmCallINTEL %[[#VoidTy]] %[[#Asm10]] %[[#Const123]] %[[#Const42]]
; CHECK: OpAsmCallINTEL %[[#VoidTy]] %[[#Asm11]] %[[#Const123]] %[[#Const42]]
; CHECK-NO: OpAsmCallINTEL

define spir_kernel void @foo(ptr addrspace(1) %_arg_int, ptr addrspace(1) %_arg_float, ptr addrspace(1) %_arg_half, i64 %_lng) {
  %i1 = load i32, ptr addrspace(1) %_arg_int
  %i2 = load i8, ptr addrspace(1) %_arg_int
  %f1 = load float, ptr addrspace(1) %_arg_float
  %h1 = load half, ptr addrspace(1) %_arg_half
  ; inline asm
  call void asm sideeffect "", ""()
  call void asm sideeffect "nop", ""()
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
  store half %res_f2, ptr addrspace(1) %_arg_half
  ; inline asm: mixed operands of different types
  call i8 asm sideeffect "cmdext $0 $3 $1 $2", "=r,r,r,r"(float %f1, i32 123, i8 %i2)
  ; inline asm: mixed integers
  %res_i2 = call i64 asm sideeffect "icmdext $0 $3 $1 $2", "=r,r,r,r"(i64 %_lng, i32 %i1, i8 %i2)
  store i64 %res_i2, ptr addrspace(1) %_arg_int
  ; inline asm: constant arguments, misc constraints
  call void asm "constcmd $0 $1", "r,r"(i32 123, double 42.0)
  call void asm "constcmd $0 $1", "i,i"(i32 123, double 42.0)
  ret void
}
