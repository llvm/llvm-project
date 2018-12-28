;;__kernel void testAtomicCompareExchangeExplicit_cl20(
;;    volatile global atomic_int* object,
;;    global int* expected,
;;    int desired)
;;{
;;  // Values of memory order and memory scope arguments correspond to SPIR-2.0 spec.
;;  atomic_compare_exchange_strong_explicit(object, expected, desired,
;;                                          memory_order_release, // 3
;;                                          memory_order_relaxed  // 0
;;                                         ); // by default, assume device scope = 2
;;  atomic_compare_exchange_strong_explicit(object, expected, desired,
;;                                          memory_order_acq_rel,   // 4
;;                                          memory_order_relaxed,   // 0
;;                                          memory_scope_work_group // 1
;;                                         );
;;  atomic_compare_exchange_weak_explicit(object, expected, desired,
;;                                        memory_order_release, // 3
;;                                        memory_order_relaxed  // 0
;;                                         ); // by default, assume device scope = 2
;;  atomic_compare_exchange_weak_explicit(object, expected, desired,
;;                                        memory_order_acq_rel,   // 4
;;                                        memory_order_relaxed,   // 0
;;                                        memory_scope_work_group // 1
;;                                       );
;;}

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

;CHECK-SPIRV: TypeInt [[int:[0-9]+]] 32 0
;; Constants below correspond to the SPIR-V spec
;CHECK-SPIRV-DAG: Constant [[int]] [[DeviceScope:[0-9]+]] 1
;CHECK-SPIRV-DAG: Constant [[int]] [[WorkgroupScope:[0-9]+]] 2
;CHECK-SPIRV-DAG: Constant [[int]] [[ReleaseMemSem:[0-9]+]] 4
;CHECK-SPIRV-DAG: Constant [[int]] [[RelaxedMemSem:[0-9]+]] 0
;CHECK-SPIRV-DAG: Constant [[int]] [[AcqRelMemSem:[0-9]+]] 8

;CHECK-SPIRV: AtomicCompareExchange {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[DeviceScope]] [[ReleaseMemSem]] [[RelaxedMemSem]]
;CHECK-SPIRV: AtomicCompareExchange {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[WorkgroupScope]] [[AcqRelMemSem]] [[RelaxedMemSem]]
;CHECK-SPIRV: AtomicCompareExchangeWeak {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[DeviceScope]] [[ReleaseMemSem]] [[RelaxedMemSem]]
;CHECK-SPIRV: AtomicCompareExchangeWeak {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[WorkgroupScope]] [[AcqRelMemSem]] [[RelaxedMemSem]]

;CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPii12memory_orderS4_12memory_scope(i32 addrspace(4)* %0, i32* %expected1, i32 %desired, i32 3, i32 0, i32 2)
;CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPii12memory_orderS4_12memory_scope(i32 addrspace(4)* %0, i32* %expected2, i32 %desired, i32 4, i32 0, i32 1)
;CHECK-LLVM: call spir_func i1 @_Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_AtomiciPii12memory_orderS4_12memory_scope(i32 addrspace(4)* %0, i32* %expected3, i32 %desired, i32 3, i32 0, i32 2)
;CHECK-LLVM: call spir_func i1 @_Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_AtomiciPii12memory_orderS4_12memory_scope(i32 addrspace(4)* %0, i32* %expected4, i32 %desired, i32 4, i32 0, i32 1)

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: nounwind
define spir_kernel void @testAtomicCompareExchangeExplicit_cl20(i32 addrspace(1)* %object, i32 addrspace(1)* %expected, i32 %desired) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %0 = addrspacecast i32 addrspace(1)* %object to i32 addrspace(4)*
  %1 = addrspacecast i32 addrspace(1)* %expected to i32 addrspace(4)*
  %call = tail call spir_func zeroext i1 @_Z39atomic_compare_exchange_strong_explicitPVU3AS4U7_AtomiciPU3AS4ii12memory_orderS4_(i32 addrspace(4)* %0, i32 addrspace(4)* %1, i32 %desired, i32 3, i32 0) #2
  %call1 = tail call spir_func zeroext i1 @_Z39atomic_compare_exchange_strong_explicitPVU3AS4U7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)* %0, i32 addrspace(4)* %1, i32 %desired, i32 4, i32 0, i32 1) #2
  %call2 = tail call spir_func zeroext i1 @_Z37atomic_compare_exchange_weak_explicitPVU3AS4U7_AtomiciPU3AS4ii12memory_orderS4_(i32 addrspace(4)* %0, i32 addrspace(4)* %1, i32 %desired, i32 3, i32 0) #2
  %call3 = tail call spir_func zeroext i1 @_Z37atomic_compare_exchange_weak_explicitPVU3AS4U7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)* %0, i32 addrspace(4)* %1, i32 %desired, i32 4, i32 0, i32 1) #2
  ret void
}

declare spir_func zeroext i1 @_Z39atomic_compare_exchange_strong_explicitPVU3AS4U7_AtomiciPU3AS4ii12memory_orderS4_(i32 addrspace(4)*, i32 addrspace(4)*, i32, i32, i32) #1

declare spir_func zeroext i1 @_Z39atomic_compare_exchange_strong_explicitPVU3AS4U7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)*, i32 addrspace(4)*, i32, i32, i32, i32) #1

declare spir_func zeroext i1 @_Z37atomic_compare_exchange_weak_explicitPVU3AS4U7_AtomiciPU3AS4ii12memory_orderS4_(i32 addrspace(4)*, i32 addrspace(4)*, i32, i32, i32) #1

declare spir_func zeroext i1 @_Z37atomic_compare_exchange_weak_explicitPVU3AS4U7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)*, i32 addrspace(4)*, i32, i32, i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}

!1 = !{i32 1, i32 1, i32 0}
!2 = !{!"none", !"none", !"none"}
!3 = !{!"atomic_int*", !"int*", !"int"}
!4 = !{!"_Atomic(int)*", !"int*", !"int"}
!5 = !{!"volatile", !"", !""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{}
