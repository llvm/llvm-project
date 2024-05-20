; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; Types:
; CHECK-DAG:  %[[#INT:]] = OpTypeInt 32
; CHECK-DAG:  %[[#BOOL:]] = OpTypeBool
;; Constants:
; CHECK-DAG:  %[[#SEQ_CONSI_SEMAN:]] = OpConstant %[[#INT]] 16
; CHECK-DAG:  %[[#DEVICE_SCOPE:]] = OpConstant %[[#INT]] 1
; CHECK-DAG:  %[[#WORKGROUP_SCOPE:]] = OpConstant %[[#INT]] 2
;; Instructions:
; CHECK-DAG:  %[[#PARAM:]] = OpFunctionParameter %[[#]]
; CHECK:      %[[#]] = OpAtomicFlagTestAndSet %[[#BOOL]] %[[#PARAM]] %[[#DEVICE_SCOPE]] %[[#SEQ_CONSI_SEMAN]]
; CHECK:      %[[#]] = OpAtomicFlagTestAndSet %[[#BOOL]] %[[#PARAM]] %[[#DEVICE_SCOPE]] %[[#SEQ_CONSI_SEMAN]]
; CHECK:      %[[#]] = OpAtomicFlagTestAndSet %[[#BOOL]] %[[#PARAM]] %[[#WORKGROUP_SCOPE]] %[[#SEQ_CONSI_SEMAN]]
; CHECK:      OpAtomicFlagClear %[[#PARAM]] %[[#DEVICE_SCOPE]] %[[#SEQ_CONSI_SEMAN]]
; CHECK:      OpAtomicFlagClear %[[#PARAM]] %[[#DEVICE_SCOPE]] %[[#SEQ_CONSI_SEMAN]]
; CHECK:      OpAtomicFlagClear %[[#PARAM]] %[[#WORKGROUP_SCOPE]] %[[#SEQ_CONSI_SEMAN]]

define spir_kernel void @testAtomicFlag(ptr %object) {
entry:
  %call1 = call spir_func zeroext i1 @_Z24atomic_flag_test_and_setPU3AS4VU7_Atomici(ptr %object)
  %call2 = call spir_func zeroext i1 @_Z33atomic_flag_test_and_set_explicitPU3AS4VU7_Atomici12memory_order(ptr %object, i32 5)
  %call5 = call spir_func zeroext i1 @_Z33atomic_flag_test_and_set_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(ptr %object, i32 5, i32 1)
  call spir_func void @_Z17atomic_flag_clearPU3AS4VU7_Atomici(ptr %object)
  call spir_func void @_Z26atomic_flag_clear_explicitPU3AS4VU7_Atomici12memory_order(ptr %object, i32 5)
  call spir_func void @_Z26atomic_flag_clear_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(ptr %object, i32 5, i32 1)
  ret void
}

declare spir_func zeroext i1 @_Z24atomic_flag_test_and_setPU3AS4VU7_Atomici(ptr)

declare spir_func zeroext i1 @_Z33atomic_flag_test_and_set_explicitPU3AS4VU7_Atomici12memory_order(ptr, i32)

declare spir_func zeroext i1 @_Z33atomic_flag_test_and_set_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(ptr, i32, i32)

declare spir_func void @_Z17atomic_flag_clearPU3AS4VU7_Atomici(ptr)

declare spir_func void @_Z26atomic_flag_clear_explicitPU3AS4VU7_Atomici12memory_order(ptr, i32)

declare spir_func void @_Z26atomic_flag_clear_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(ptr, i32, i32)
