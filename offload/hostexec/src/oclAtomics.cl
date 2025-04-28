
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define __atomic_ulong atomic_ulong
#define __inl __attribute__((flatten, always_inline))

// mem order codes: A=acquire, X=relaxed, R=release
extern __inl ulong oclAtomic64Load_A(__global __atomic_ulong * Address){
  return  __opencl_atomic_load(Address, memory_order_acquire, memory_scope_all_svm_devices);
}
extern __inl ulong oclAtomic64Load_X(__global __atomic_ulong * Address){
  return  __opencl_atomic_load(Address, memory_order_relaxed, memory_scope_all_svm_devices);
}
extern __inl uint oclAtomic32Load_A(__global const atomic_uint * Address){
  return  __opencl_atomic_load(Address, memory_order_acquire, memory_scope_all_svm_devices);
}
extern __inl uint oclAtomic64CAS_AX(__global __atomic_ulong * Address,  ulong * e_val, ulong new_val) {
   return __opencl_atomic_compare_exchange_strong( Address, e_val, new_val,
     memory_order_acquire, memory_order_relaxed, memory_scope_all_svm_devices);
}
extern __inl uint oclAtomic64CAS_RX(__global __atomic_ulong * Address,  ulong * e_val, ulong new_val) {
   return __opencl_atomic_compare_exchange_strong(Address, e_val, new_val,
     memory_order_release, memory_order_relaxed, memory_scope_all_svm_devices);
}
#undef __atomic_ulong
#undef __inl
