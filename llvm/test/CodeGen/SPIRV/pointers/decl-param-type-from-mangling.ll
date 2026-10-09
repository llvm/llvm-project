; Pointer parameters of non-builtin external declarations get their pointee
; type from the mangled name instead of defaulting to i8.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#Load:]] "_Z8ext_loadPKm"
; CHECK-DAG: OpName %[[#Store:]] "_Z9ext_storePKmm"
; CHECK-DAG: OpName %[[#LoadV4:]] "_Z10ext_loadv4PKDv4_f"
; CHECK-DAG: OpName %[[#LongLong:]] "_Z12ext_longlongPy"
; CHECK-DAG: OpName %[[#Ref:]] "_Z7ext_refRKi"
; CHECK-DAG: OpName %[[#AS:]] "_Z6ext_asPU3AS1Kf"
; CHECK-DAG: OpName %[[#PtrPtr:]] "_Z10ext_ptrptrPPi"
; CHECK-DAG: OpName %[[#VoidPtr:]] "_Z11ext_voidptrPv"
; CHECK-DAG: OpName %[[#BoolPtr:]] "_Z11ext_boolptrPb"
; CHECK-DAG: OpName %[[#Struct:]] "_Z10ext_structP7intlist"
; CHECK-DAG: OpName %[[#FnPtr:]] "_Z9ext_fnptrPFiiE"
; CHECK-DAG: OpName %[[#Mismatch:]] "_Z12ext_mismatchPKi"

; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Long:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#V4Float:]] = OpTypeVector %[[#Float]] 4
; CHECK-DAG: %[[#GenPtrLong:]] = OpTypePointer Generic %[[#Long]]
; CHECK-DAG: %[[#GenPtrV4Float:]] = OpTypePointer Generic %[[#V4Float]]
; CHECK-DAG: %[[#GenPtrChar:]] = OpTypePointer Generic %[[#Char]]
; CHECK-DAG: %[[#GenPtrInt:]] = OpTypePointer Generic %[[#Int]]
; CHECK-DAG: %[[#CWPtrLong:]] = OpTypePointer CrossWorkgroup %[[#Long]]
; CHECK-DAG: %[[#CWPtrInt:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; CHECK-DAG: %[[#CWPtrFloat:]] = OpTypePointer CrossWorkgroup %[[#Float]]

; CHECK-DAG: %[[#LongPtrLongTy:]] = OpTypeFunction %[[#Long]] %[[#GenPtrLong]]
; CHECK-DAG: %[[#StoreTy:]] = OpTypeFunction %[[#Void]] %[[#GenPtrLong]] %[[#Long]]
; CHECK-DAG: %[[#LoadV4Ty:]] = OpTypeFunction %[[#V4Float]] %[[#GenPtrV4Float]]
; CHECK-DAG: %[[#IntPtrIntTy:]] = OpTypeFunction %[[#Int]] %[[#GenPtrInt]]
; The storage class comes from the IR address space, not from the mangling.
; CHECK-DAG: %[[#ASTy:]] = OpTypeFunction %[[#Float]] %[[#CWPtrFloat]]
; Pointer-to-pointer, void*, bool*, aggregates and function pointers keep the
; i8 default.
; CHECK-DAG: %[[#IntPtrCharTy:]] = OpTypeFunction %[[#Int]] %[[#GenPtrChar]]
; CHECK-DAG: %[[#VoidPtrCharTy:]] = OpTypeFunction %[[#Void]] %[[#GenPtrChar]]

; CHECK: %[[#Load]] = OpFunction %[[#Long]] None %[[#LongPtrLongTy]]
; CHECK: OpFunctionParameter %[[#GenPtrLong]]
; CHECK: %[[#Store]] = OpFunction %[[#Void]] None %[[#StoreTy]]
; CHECK: OpFunctionParameter %[[#GenPtrLong]]
; CHECK: %[[#LoadV4]] = OpFunction %[[#V4Float]] None %[[#LoadV4Ty]]
; CHECK: OpFunctionParameter %[[#GenPtrV4Float]]
; CHECK: %[[#LongLong]] = OpFunction %[[#Long]] None %[[#LongPtrLongTy]]
; CHECK: OpFunctionParameter %[[#GenPtrLong]]
; CHECK: %[[#Ref]] = OpFunction %[[#Int]] None %[[#IntPtrIntTy]]
; CHECK: OpFunctionParameter %[[#GenPtrInt]]
; CHECK: %[[#AS]] = OpFunction %[[#Float]] None %[[#ASTy]]
; CHECK: OpFunctionParameter %[[#CWPtrFloat]]
; CHECK: %[[#PtrPtr]] = OpFunction %[[#Int]] None %[[#IntPtrCharTy]]
; CHECK: OpFunctionParameter %[[#GenPtrChar]]
; CHECK: %[[#VoidPtr]] = OpFunction %[[#Void]] None %[[#VoidPtrCharTy]]
; CHECK: OpFunctionParameter %[[#GenPtrChar]]
; CHECK: %[[#BoolPtr]] = OpFunction %[[#Int]] None %[[#IntPtrCharTy]]
; CHECK: OpFunctionParameter %[[#GenPtrChar]]
; CHECK: %[[#Struct]] = OpFunction %[[#Int]] None %[[#IntPtrCharTy]]
; CHECK: OpFunctionParameter %[[#GenPtrChar]]
; CHECK: %[[#FnPtr]] = OpFunction %[[#Int]] None %[[#IntPtrCharTy]]
; CHECK: OpFunctionParameter %[[#GenPtrChar]]
; CHECK: %[[#Mismatch]] = OpFunction %[[#Int]] None %[[#IntPtrIntTy]]
; CHECK: OpFunctionParameter %[[#GenPtrInt]]

; CHECK: OpFunction %[[#Void]] None %[[#]]
; CHECK: %[[#In:]] = OpFunctionParameter %[[#CWPtrLong]]
; CHECK: %[[#Out:]] = OpFunctionParameter %[[#CWPtrLong]]
; CHECK: %[[#F:]] = OpFunctionParameter %[[#CWPtrFloat]]
; CHECK: %[[#FI:]] = OpBitcast %[[#CWPtrInt]] %[[#F]]
; CHECK: %[[#InG:]] = OpPtrCastToGeneric %[[#GenPtrLong]] %[[#In]]
; CHECK-NEXT: OpFunctionCall %[[#Long]] %[[#Load]] %[[#InG]]
; CHECK: %[[#OutG:]] = OpPtrCastToGeneric %[[#GenPtrLong]] %[[#Out]]
; CHECK-NEXT: OpFunctionCall %[[#Void]] %[[#Store]] %[[#OutG]] %[[#]]
; CHECK: %[[#V4G:]] = OpPtrCastToGeneric %[[#GenPtrV4Float]] %[[#]]
; CHECK-NEXT: OpFunctionCall %[[#V4Float]] %[[#LoadV4]] %[[#V4G]]
; CHECK: OpFunctionCall %[[#Long]] %[[#LongLong]] %[[#InG]]
; CHECK: %[[#FG:]] = OpPtrCastToGeneric %[[#GenPtrInt]] %[[#FI]]
; CHECK-NEXT: OpFunctionCall %[[#Int]] %[[#Ref]] %[[#FG]]
; CHECK: OpFunctionCall %[[#Float]] %[[#AS]] %[[#F]]
; CHECK: OpFunctionCall %[[#Int]] %[[#PtrPtr]]
; CHECK: OpFunctionCall %[[#Void]] %[[#VoidPtr]]
; CHECK: OpFunctionCall %[[#Int]] %[[#BoolPtr]]
; CHECK: OpFunctionCall %[[#Int]] %[[#Struct]]
; CHECK: OpFunctionCall %[[#Int]] %[[#FnPtr]]
; CHECK: OpFunctionCall %[[#Int]] %[[#Mismatch]] %[[#FG]]

; ulong ext_load(const ulong *p)
declare spir_func i64 @_Z8ext_loadPKm(ptr addrspace(4))
; void ext_store(const ulong *p, ulong v)
declare spir_func void @_Z9ext_storePKmm(ptr addrspace(4), i64)
; float4 ext_loadv4(const float4 *p)
declare spir_func <4 x float> @_Z10ext_loadv4PKDv4_f(ptr addrspace(4))
; unsigned long long ext_longlong(unsigned long long *p)
declare spir_func i64 @_Z12ext_longlongPy(ptr addrspace(4))
; int ext_ref(const int &p)
declare spir_func i32 @_Z7ext_refRKi(ptr addrspace(4))
; float ext_as(const __global float *p)
declare spir_func float @_Z6ext_asPU3AS1Kf(ptr addrspace(1))
; int ext_ptrptr(int **p)
declare spir_func i32 @_Z10ext_ptrptrPPi(ptr addrspace(4))
; void ext_voidptr(void *p)
declare spir_func void @_Z11ext_voidptrPv(ptr addrspace(4))
; int ext_boolptr(bool *p)
declare spir_func i32 @_Z11ext_boolptrPb(ptr addrspace(4))
; int ext_struct(struct intlist *p)
declare spir_func i32 @_Z10ext_structP7intlist(ptr addrspace(4))
; int ext_fnptr(int (*p)(int))
declare spir_func i32 @_Z9ext_fnptrPFiiE(ptr addrspace(4))
; int ext_mismatch(const int *p)
declare spir_func i32 @_Z12ext_mismatchPKi(ptr addrspace(4))

define spir_kernel void @test(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %v4, ptr addrspace(1) %pp, ptr addrspace(1) %vp, ptr addrspace(1) %f) {
entry:
  %in.g = addrspacecast ptr addrspace(1) %in to ptr addrspace(4)
  %v = call spir_func i64 @_Z8ext_loadPKm(ptr addrspace(4) %in.g)
  %v1 = add i64 %v, 1
  %out.g = addrspacecast ptr addrspace(1) %out to ptr addrspace(4)
  call spir_func void @_Z9ext_storePKmm(ptr addrspace(4) %out.g, i64 %v1)
  %v4.g = addrspacecast ptr addrspace(1) %v4 to ptr addrspace(4)
  %x = call spir_func <4 x float> @_Z10ext_loadv4PKDv4_f(ptr addrspace(4) %v4.g)
  %ll = call spir_func i64 @_Z12ext_longlongPy(ptr addrspace(4) %in.g)
  %fv = load float, ptr addrspace(1) %f
  %f.g = addrspacecast ptr addrspace(1) %f to ptr addrspace(4)
  %r = call spir_func i32 @_Z7ext_refRKi(ptr addrspace(4) %f.g)
  %a = call spir_func float @_Z6ext_asPU3AS1Kf(ptr addrspace(1) %f)
  %pp.g = addrspacecast ptr addrspace(1) %pp to ptr addrspace(4)
  %y = call spir_func i32 @_Z10ext_ptrptrPPi(ptr addrspace(4) %pp.g)
  %vp.g = addrspacecast ptr addrspace(1) %vp to ptr addrspace(4)
  call spir_func void @_Z11ext_voidptrPv(ptr addrspace(4) %vp.g)
  %b = call spir_func i32 @_Z11ext_boolptrPb(ptr addrspace(4) %vp.g)
  %s = call spir_func i32 @_Z10ext_structP7intlist(ptr addrspace(4) %vp.g)
  %p = call spir_func i32 @_Z9ext_fnptrPFiiE(ptr addrspace(4) %vp.g)
  %z = call spir_func i32 @_Z12ext_mismatchPKi(ptr addrspace(4) %f.g)
  ret void
}
