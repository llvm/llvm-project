; RUN: llc -O0 -mtriple=spirv32v1.3-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.3-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; __kernel void testAtomicCompareExchangeExplicit_cl20(
;;     volatile global atomic_int* object,
;;     global int* expected,
;;     int desired)
;; {
  ;; Values of memory order and memory scope arguments correspond to SPIR-2.0 spec.
;;   atomic_compare_exchange_strong_explicit(object, expected, desired,
;;                                           memory_order_release, // 3
;;                                           memory_order_relaxed  // 0
;;                                          ); // by default, assume device scope = 2
;;   atomic_compare_exchange_strong_explicit(object, expected, desired,
;;                                           memory_order_acq_rel,   // 4
;;                                           memory_order_relaxed,   // 0
;;                                           memory_scope_work_group // 1
;;                                          );
;;   atomic_compare_exchange_weak_explicit(object, expected, desired,
;;                                         memory_order_release, // 3
;;                                         memory_order_relaxed  // 0
;;                                          ); // by default, assume device scope = 2
;;   atomic_compare_exchange_weak_explicit(object, expected, desired,
;;                                         memory_order_acq_rel,   // 4
;;                                         memory_order_relaxed,   // 0
;;                                         memory_scope_work_group // 1
;;                                        );
;; }

; CHECK-SPIRV: %[[#int:]] = OpTypeInt 32 0
;; Constants below correspond to the SPIR-V spec
; CHECK-SPIRV-DAG: %[[#DeviceScope:]] = OpConstant %[[#int]] 1
; CHECK-SPIRV-DAG: %[[#WorkgroupScope:]] = OpConstant %[[#int]] 2
; CHECK-SPIRV-DAG: %[[#ReleaseMemSem:]] = OpConstant %[[#int]] 4
; CHECK-SPIRV-DAG: %[[#RelaxedMemSem:]] = OpConstant %[[#int]] 0
; CHECK-SPIRV-DAG: %[[#AcqRelMemSem:]] = OpConstant %[[#int]] 8

; CHECK-SPIRV: %[[#]] = OpAtomicCompareExchange %[[#]] %[[#]] %[[#DeviceScope]] %[[#ReleaseMemSem]] %[[#RelaxedMemSem]]
; CHECK-SPIRV: %[[#]] = OpAtomicCompareExchange %[[#]] %[[#]] %[[#WorkgroupScope]] %[[#AcqRelMemSem]] %[[#RelaxedMemSem]]
; CHECK-SPIRV: %[[#]] = OpAtomicCompareExchangeWeak %[[#]] %[[#]] %[[#DeviceScope]] %[[#ReleaseMemSem]] %[[#RelaxedMemSem]]
; CHECK-SPIRV: %[[#]] = OpAtomicCompareExchangeWeak %[[#]] %[[#]] %[[#WorkgroupScope]] %[[#AcqRelMemSem]] %[[#RelaxedMemSem]]

define dso_local spir_kernel void @testAtomicCompareExchangeExplicit_cl20(i32 addrspace(1)* noundef %object, i32 addrspace(1)* noundef %expected, i32 noundef %desired) local_unnamed_addr {
entry:
  %0 = addrspacecast i32 addrspace(1)* %object to i32 addrspace(4)*
  %1 = addrspacecast i32 addrspace(1)* %expected to i32 addrspace(4)*
  %call = call spir_func zeroext i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_(i32 addrspace(4)* noundef %0, i32 addrspace(4)* noundef %1, i32 noundef %desired, i32 noundef 3, i32 noundef 0)
  %call1 = call spir_func zeroext i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)* noundef %0, i32 addrspace(4)* noundef %1, i32 noundef %desired, i32 noundef 4, i32 noundef 0, i32 noundef 1)
  %call2 = call spir_func zeroext i1 @_Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_(i32 addrspace(4)* noundef %0, i32 addrspace(4)* noundef %1, i32 noundef %desired, i32 noundef 3, i32 noundef 0)
  %call3 = call spir_func zeroext i1 @_Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)* noundef %0, i32 addrspace(4)* noundef %1, i32 noundef %desired, i32 noundef 4, i32 noundef 0, i32 noundef 1)
  ret void
}

declare spir_func zeroext i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_(i32 addrspace(4)* noundef, i32 addrspace(4)* noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr

declare spir_func zeroext i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)* noundef, i32 addrspace(4)* noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr

declare spir_func zeroext i1 @_Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_(i32 addrspace(4)* noundef, i32 addrspace(4)* noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr

declare spir_func zeroext i1 @_Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)* noundef, i32 addrspace(4)* noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr
