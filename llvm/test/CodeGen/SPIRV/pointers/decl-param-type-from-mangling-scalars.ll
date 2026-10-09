; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Cover unambiguous scalar spellings and combined restrict/volatile qualifiers.
; CHECK-DAG: OpName %[[#Char:]] "_Z4takePa"
; CHECK-DAG: OpName %[[#Short:]] "_Z4takePs"
; CHECK-DAG: OpName %[[#Double:]] "_Z4takePd"
; CHECK-DAG: OpName %[[#LongLong:]] "_Z4takePx"
; CHECK-DAG: OpName %[[#Qual:]] "_Z4takePVri"
; CHECK-DAG: OpName %[[#Half:]] "_Z4takePDF16_"
; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#CharTy:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#CharPtr:]] = OpTypePointer CrossWorkgroup %[[#CharTy]]
; CHECK-DAG: %[[#CharFn:]] = OpTypeFunction %[[#Void]] %[[#CharPtr]]
; CHECK-DAG: %[[#ShortTy:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#ShortPtr:]] = OpTypePointer CrossWorkgroup %[[#ShortTy]]
; CHECK-DAG: %[[#ShortFn:]] = OpTypeFunction %[[#Void]] %[[#ShortPtr]]
; CHECK-DAG: %[[#DoubleTy:]] = OpTypeFloat 64
; CHECK-DAG: %[[#DoublePtr:]] = OpTypePointer CrossWorkgroup %[[#DoubleTy]]
; CHECK-DAG: %[[#DoubleFn:]] = OpTypeFunction %[[#Void]] %[[#DoublePtr]]
; CHECK-DAG: %[[#LongLongTy:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#LongLongPtr:]] = OpTypePointer CrossWorkgroup %[[#LongLongTy]]
; CHECK-DAG: %[[#LongLongFn:]] = OpTypeFunction %[[#Void]] %[[#LongLongPtr]]
; CHECK-DAG: %[[#QualTy:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#QualPtr:]] = OpTypePointer CrossWorkgroup %[[#QualTy]]
; CHECK-DAG: %[[#QualFn:]] = OpTypeFunction %[[#Void]] %[[#QualPtr]]
; CHECK-DAG: %[[#HalfTy:]] = OpTypeFloat 16
; CHECK-DAG: %[[#HalfPtr:]] = OpTypePointer CrossWorkgroup %[[#HalfTy]]
; CHECK-DAG: %[[#HalfFn:]] = OpTypeFunction %[[#Void]] %[[#HalfPtr]]

; CHECK: %[[#Char]] = OpFunction %[[#Void]] None %[[#CharFn]]
; CHECK: %[[#Short]] = OpFunction %[[#Void]] None %[[#ShortFn]]
; CHECK: %[[#Double]] = OpFunction %[[#Void]] None %[[#DoubleFn]]
; CHECK: %[[#LongLong]] = OpFunction %[[#Void]] None %[[#LongLongFn]]
; CHECK: %[[#Qual]] = OpFunction %[[#Void]] None %[[#QualFn]]
; CHECK: %[[#Half]] = OpFunction %[[#Void]] None %[[#HalfFn]]

declare spir_func void @_Z4takePa(ptr addrspace(1))
declare spir_func void @_Z4takePs(ptr addrspace(1))
declare spir_func void @_Z4takePd(ptr addrspace(1))
declare spir_func void @_Z4takePx(ptr addrspace(1))
declare spir_func void @_Z4takePVri(ptr addrspace(1))
declare spir_func void @_Z4takePDF16_(ptr addrspace(1))

define spir_kernel void @test(ptr addrspace(1) %p) {
  call spir_func void @_Z4takePa(ptr addrspace(1) %p)
  call spir_func void @_Z4takePs(ptr addrspace(1) %p)
  call spir_func void @_Z4takePd(ptr addrspace(1) %p)
  call spir_func void @_Z4takePx(ptr addrspace(1) %p)
  call spir_func void @_Z4takePVri(ptr addrspace(1) %p)
  call spir_func void @_Z4takePDF16_(ptr addrspace(1) %p)
  ret void
}
