; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_INTEL_inline_assembly -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_INTEL_inline_assembly -o - -filetype=obj | spirv-val %}

; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: Inline assembly instructions require the following SPIR-V extension: SPV_INTEL_inline_assembly

; CHECK: OpCapability AsmINTEL
; CHECK: OpExtension "SPV_INTEL_inline_assembly"

; CHECK-COUNT-11: OpDecorate %[[#]] SideEffectsINTEL

; CHECK-DAG: %[[#VoidTy:]] = OpTypeVoid
; CHECK-DAG: %[[#Int8Ty:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Int64Ty:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#HalfTy:]] = OpTypeFloat 16
; CHECK-DAG: %[[#FloatTy:]] = OpTypeFloat 32
; CHECK-DAG: %[[#DoubleTy:]] = OpTypeFloat 64
; CHECK-DAG: %[[#StructTy:]] = OpTypeStruct %[[#Int32Ty]] %[[#FloatTy]] %[[#HalfTy]]
; CHECK-DAG: %[[#StructTy1:]] = OpTypeStruct %[[#Int32Ty]] %[[#FloatTy]]
; CHECK-DAG: %[[#Int8PtrTy:]] = OpTypePointer CrossWorkgroup %[[#Int8Ty]]

; CHECK-DAG: OpTypeFunction %[[#VoidTy]] %[[#]] %[[#]] %[[#]] %[[#Int64Ty]]
; CHECK-DAG: %[[#Fun1Ty:]] = OpTypeFunction %[[#VoidTy]]
; CHECK-DAG: %[[#Fun2Ty:]] = OpTypeFunction %[[#Int32Ty]]
; CHECK-DAG: %[[#Fun3Ty:]] = OpTypeFunction %[[#Int32Ty]] %[[#Int32Ty]]
; CHECK-DAG: %[[#Fun4Ty:]] = OpTypeFunction %[[#FloatTy]] %[[#FloatTy]]
; CHECK-DAG: %[[#Fun5Ty:]] = OpTypeFunction %[[#HalfTy]] %[[#FloatTy]] %[[#FloatTy]]
; CHECK-DAG: %[[#Fun6Ty:]] = OpTypeFunction %[[#Int8Ty]] %[[#FloatTy]] %[[#Int32Ty]] %[[#Int8Ty]]
; CHECK-DAG: %[[#Fun7Ty:]] = OpTypeFunction %[[#Int64Ty]] %[[#Int64Ty]] %[[#Int32Ty]] %[[#Int8Ty]]
; CHECK-DAG: %[[#Fun8Ty:]] = OpTypeFunction %[[#VoidTy]] %[[#Int32Ty]] %[[#DoubleTy]]
; CHECK-DAG: %[[#Fun9Ty:]] = OpTypeFunction %[[#StructTy]] %[[#Int32Ty]] %[[#FloatTy]] %[[#HalfTy]]
; CHECK-DAG: %[[#Fun10Ty:]] = OpTypeFunction %[[#Int32Ty]] %[[#Int8PtrTy]] %[[#Int32Ty]] %[[#Int8PtrTy]]
; CHECK-DAG: %[[#Fun11Ty:]] = OpTypeFunction %[[#StructTy1]] %[[#Int8PtrTy]] %[[#Int32Ty]] %[[#FloatTy]]

; CHECK-DAG: %[[#Const2:]] = OpConstant %[[#FloatTy]] 2
; CHECK-DAG: %[[#Const123:]] = OpConstant %[[#Int32Ty]] 123
; CHECK-DAG: %[[#Const42:]] = OpConstant %[[#DoubleTy:]] 42

; CHECK-DAG: %[[#Dialect:]] = OpAsmTargetINTEL "spirv64-unknown-unknown"
; CHECK-NO: OpAsmTargetINTEL

; CHECK-DAG: %[[#Asm1:]] = OpAsmINTEL %[[#VoidTy]] %[[#Fun1Ty]] %[[#Dialect]] "" ""
; CHECK-DAG: %[[#Asm2:]] = OpAsmINTEL %[[#VoidTy]] %[[#Fun1Ty]] %[[#Dialect]] "nop" ""
; CHECK-DAG: %[[#Asm3:]] = OpAsmINTEL %[[#VoidTy]] %[[#Fun1Ty]] %[[#Dialect]] "" "~{cc},~{memory}"
; CHECK-DAG: %[[#Asm4:]] = OpAsmINTEL %[[#Int32Ty]] %[[#Fun2Ty:]] %[[#Dialect]] "clobber_out $0" "=&r"
; CHECK-DAG: %[[#Asm5:]] = OpAsmINTEL %[[#Int32Ty]] %[[#Fun3Ty]] %[[#Dialect]] "icmd $0 $1" "=r,r"
; CHECK-DAG: %[[#Asm6:]] = OpAsmINTEL %[[#FloatTy]] %[[#Fun4Ty]] %[[#Dialect]] "fcmd $0 $1" "=r,r"
; CHECK-DAG: %[[#Asm7:]] = OpAsmINTEL %[[#HalfTy]] %[[#Fun5Ty]] %[[#Dialect]] "fcmdext $0 $1 $2" "=r,r,r"
; CHECK-DAG: %[[#Asm8:]] = OpAsmINTEL %[[#Int8Ty]] %[[#Fun6Ty]] %[[#Dialect]] "cmdext $0 $3 $1 $2" "=r,r,r,r"
; CHECK-DAG: %[[#Asm9:]] = OpAsmINTEL %[[#Int64Ty]] %[[#Fun7Ty]] %[[#Dialect]] "icmdext $0 $3 $1 $2" "=r,r,r,r"
; CHECK-DAG: %[[#Asm10:]] = OpAsmINTEL %[[#VoidTy]] %[[#Fun8Ty]] %[[#Dialect]] "constcmd $0 $1" "r,r"
; CHECK-DAG: %[[#Asm11:]] = OpAsmINTEL %[[#VoidTy]] %[[#Fun8Ty]] %[[#Dialect]] "constcmd $0 $1" "i,i"
; CHECK-DAG: %[[#Asm12:]] = OpAsmINTEL %[[#StructTy]] %[[#Fun9Ty]] %[[#Dialect]] "cmdext $0 $4 $5\n cmdext $1 $5 $6\n cmdext $2 $4 $6" "=&r,=&r,=&r,r,r,r"
; CHECK-DAG: %[[#Asm13:]] = OpAsmINTEL %[[#Int32Ty]] %[[#Fun10Ty]] %[[#Dialect]] "icmdext $0 $2 $3"
; CHECK-DAG: %[[#Asm14:]] = OpAsmINTEL %[[#StructTy1]] %[[#Fun11Ty]] %[[#Dialect]] "cmdext $0 $3 $4\n cmdext $1 $3 $4\n $2 $3 $4" "=r,=&r,=*m, r, r"
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
; CHECK: %[[#StructRet:]] = OpAsmCallINTEL %[[#StructTy]] %[[#Asm12]]
; CHECK-NEXT: OpCompositeExtract %[[#Int32Ty]] %[[#StructRet]] 0
; CHECK-NEXT: OpCompositeExtract %[[#FloatTy]] %[[#StructRet]] 1
; CHECK-NEXT: OpCompositeExtract %[[#HalfTy]] %[[#StructRet]] 2
; CHECK: OpAsmCallINTEL %[[#Int32Ty]] %[[#Asm13]]
; CHECK: %[[#StructRet1:]] = OpAsmCallINTEL %[[#StructTy1]] %[[#Asm14]]
; CHECK-NEXT: OpCompositeExtract %[[#Int32Ty]] %[[#StructRet1]] 0
; CHECK-NEXT: OpCompositeExtract %[[#FloatTy]] %[[#StructRet1]] 1
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
  ; inline asm: multiple outputs, hence aggregate return
  %res_struct = call { i32, float, half } asm sideeffect "cmdext $0 $4 $5\0A cmdext $1 $5 $6\0A cmdext $2 $4 $6", "=&r,=&r,=&r,r,r,r"(i32 %i1, float %f1, half %h1)
  %asmresult = extractvalue { i32, float, half } %res_struct, 0
  %asmresult3 = extractvalue { i32, float, half } %res_struct, 1
  %asmresult4 = extractvalue { i32, float, half } %res_struct, 2
  store i32 %asmresult, ptr addrspace(1) %_arg_int
  store float %asmresult3, ptr addrspace(1) %_arg_float
  store half %asmresult4, ptr addrspace(1) %_arg_half
  ; inline asm: two outputs but one is a mem operand
  %asmtmp = call i32 asm sideeffect "icmdext $0 $2 $3", "=&r,=*m,r,*m,~{memory}"(ptr addrspace(1) elementtype(i32) %_arg_int, i32 %i1, ptr addrspace(1) elementtype(float) %_arg_float)
  store i32 %asmtmp, ptr addrspace(1) %_arg_int
  ; inline asm: multiple outputs out of which one is a mem operand
  %res_struct1 = call { i32, float } asm sideeffect "cmdext $0 $3 $4\0A cmdext $1 $3 $4\0A $2 $3 $4", "=r,=&r,=*m, r, r"(ptr addrspace(1) elementtype(i32) %_arg_int, i32 %i1, float %f1)
  %asmresult1 = extractvalue { i32, float } %res_struct1, 0
  %asmresult2 = extractvalue { i32, float } %res_struct1, 1
  store i32 %asmresult1, ptr addrspace(1) %_arg_int
  store float %asmresult2, ptr addrspace(1) %_arg_float
  ret void
}
