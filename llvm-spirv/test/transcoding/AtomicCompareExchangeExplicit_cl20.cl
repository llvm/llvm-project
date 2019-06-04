// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
// RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

__kernel void testAtomicCompareExchangeExplicit_cl20(
    volatile global atomic_int* object,
    global int* expected,
    int desired)
{
  // Values of memory order and memory scope arguments correspond to SPIR-2.0 spec.
  atomic_compare_exchange_strong_explicit(object, expected, desired,
                                          memory_order_release, // 3
                                          memory_order_relaxed  // 0
                                         ); // by default, assume device scope = 2
  atomic_compare_exchange_strong_explicit(object, expected, desired,
                                          memory_order_acq_rel,   // 4
                                          memory_order_relaxed,   // 0
                                          memory_scope_work_group // 1
                                         );
  atomic_compare_exchange_weak_explicit(object, expected, desired,
                                        memory_order_release, // 3
                                        memory_order_relaxed  // 0
                                         ); // by default, assume device scope = 2
  atomic_compare_exchange_weak_explicit(object, expected, desired,
                                        memory_order_acq_rel,   // 4
                                        memory_order_relaxed,   // 0
                                        memory_scope_work_group // 1
                                       );
}

//CHECK-SPIRV: TypeInt [[int:[0-9]+]] 32 0
//; Constants below correspond to the SPIR-V spec
//CHECK-SPIRV-DAG: Constant [[int]] [[DeviceScope:[0-9]+]] 1
//CHECK-SPIRV-DAG: Constant [[int]] [[WorkgroupScope:[0-9]+]] 2
//CHECK-SPIRV-DAG: Constant [[int]] [[ReleaseMemSem:[0-9]+]] 4
//CHECK-SPIRV-DAG: Constant [[int]] [[RelaxedMemSem:[0-9]+]] 0
//CHECK-SPIRV-DAG: Constant [[int]] [[AcqRelMemSem:[0-9]+]] 8

//CHECK-SPIRV: AtomicCompareExchange {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[DeviceScope]] [[ReleaseMemSem]] [[RelaxedMemSem]]
//CHECK-SPIRV: AtomicCompareExchange {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[WorkgroupScope]] [[AcqRelMemSem]] [[RelaxedMemSem]]
//CHECK-SPIRV: AtomicCompareExchangeWeak {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[DeviceScope]] [[ReleaseMemSem]] [[RelaxedMemSem]]
//CHECK-SPIRV: AtomicCompareExchangeWeak {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[WorkgroupScope]] [[AcqRelMemSem]] [[RelaxedMemSem]]

//CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)* %0, i32 addrspace(4)* %expected1.as, i32 %desired, i32 3, i32 0, i32 2)
//CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)* %0, i32 addrspace(4)* %expected2.as, i32 %desired, i32 4, i32 0, i32 1)
//CHECK-LLVM: call spir_func i1 @_Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)* %0, i32 addrspace(4)* %expected3.as, i32 %desired, i32 3, i32 0, i32 2)
//CHECK-LLVM: call spir_func i1 @_Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)* %0, i32 addrspace(4)* %expected4.as, i32 %desired, i32 4, i32 0, i32 1)
