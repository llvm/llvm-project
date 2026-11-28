// RUN: mlir-opt %s -convert-spirv-to-llvm -verify-diagnostics -split-input-file
// RUN: mlir-opt %s -convert-spirv-to-llvm='allow-pattern-rollback=0' -verify-diagnostics -split-input-file

// QueueFamily has no LLVM syncscope.
spirv.func @atomic_queue_family_scope(%ptr: !spirv.ptr<i32, CrossWorkgroup>, %value: i32) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.AtomicIAdd' that was explicitly marked illegal}}
  %0 = spirv.AtomicIAdd <QueueFamily> <None> %ptr, %value : !spirv.ptr<i32, CrossWorkgroup>
  spirv.Return
}

// -----

// An atomic llvm.load cannot have release or acq_rel ordering.
spirv.func @atomic_load_release(%ptr: !spirv.ptr<i32, CrossWorkgroup>) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.AtomicLoad' that was explicitly marked illegal}}
  %0 = spirv.AtomicLoad <Device> <Release> %ptr : !spirv.ptr<i32, CrossWorkgroup>
  spirv.Return
}

// -----

// An atomic llvm.store cannot have acquire ordering.
spirv.func @atomic_store_acquire(%ptr: !spirv.ptr<i32, CrossWorkgroup>, %value: i32) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.AtomicStore' that was explicitly marked illegal}}
  spirv.AtomicStore <Device> <Acquire> %ptr, %value : !spirv.ptr<i32, CrossWorkgroup>
  spirv.Return
}

// -----

// The llvm.cmpxchg failure ordering cannot be release or acq_rel.
spirv.func @atomic_compare_exchange_unequal_release(%ptr: !spirv.ptr<i32, CrossWorkgroup>, %value: i32, %comparator: i32) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.AtomicCompareExchange' that was explicitly marked illegal}}
  %0 = spirv.AtomicCompareExchange <Device> <Release> <Release> %ptr, %value, %comparator : !spirv.ptr<i32, CrossWorkgroup>
  spirv.Return
}

// -----

spirv.func @atomic_compare_exchange_unequal_acq_rel(%ptr: !spirv.ptr<i32, CrossWorkgroup>, %value: i32, %comparator: i32) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.AtomicCompareExchangeWeak' that was explicitly marked illegal}}
  %0 = spirv.AtomicCompareExchangeWeak <Device> <AcquireRelease> <AcquireRelease> %ptr, %value, %comparator : !spirv.ptr<i32, CrossWorkgroup>
  spirv.Return
}

// -----

spirv.func @atomic_load_acq_rel(%ptr: !spirv.ptr<i32, CrossWorkgroup>) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.AtomicLoad' that was explicitly marked illegal}}
  %0 = spirv.AtomicLoad <Device> <AcquireRelease> %ptr : !spirv.ptr<i32, CrossWorkgroup>
  spirv.Return
}

// -----

spirv.func @atomic_load_queue_family_scope(%ptr: !spirv.ptr<i32, CrossWorkgroup>) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.AtomicLoad' that was explicitly marked illegal}}
  %0 = spirv.AtomicLoad <QueueFamily> <None> %ptr : !spirv.ptr<i32, CrossWorkgroup>
  spirv.Return
}

// -----

spirv.func @atomic_compare_exchange_queue_family_scope(%ptr: !spirv.ptr<i32, CrossWorkgroup>, %value: i32, %comparator: i32) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.AtomicCompareExchange' that was explicitly marked illegal}}
  %0 = spirv.AtomicCompareExchange <QueueFamily> <None> <None> %ptr, %value, %comparator : !spirv.ptr<i32, CrossWorkgroup>
  spirv.Return
}
