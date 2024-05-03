// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.AtomicAnd
//===----------------------------------------------------------------------===//

func.func @atomic_and(%ptr : !spirv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spirv.AtomicAnd <Device> <None> %{{.*}}, %{{.*}} : !spirv.ptr<i32, StorageBuffer>
  %0 = spirv.AtomicAnd <Device> <None> %ptr, %value : !spirv.ptr<i32, StorageBuffer>
  return %0 : i32
}

// -----

func.func @atomic_and(%ptr : !spirv.ptr<f32, StorageBuffer>, %value : i32) -> i32 {
  // expected-error @+1 {{'spirv.AtomicAnd' op failed to verify that `value` type matches pointee type of `pointer`}}
  %0 = "spirv.AtomicAnd"(%ptr, %value) {memory_scope = #spirv.scope<Workgroup>, semantics = #spirv.memory_semantics<AcquireRelease>} : (!spirv.ptr<f32, StorageBuffer>, i32) -> (i32)
  return %0 : i32
}


// -----

func.func @atomic_and(%ptr : !spirv.ptr<i32, StorageBuffer>, %value : i64) -> i64 {
  // expected-error @+1 {{'spirv.AtomicAnd' op failed to verify that `value` type matches pointee type of `pointer`}}
  %0 = "spirv.AtomicAnd"(%ptr, %value) {memory_scope = #spirv.scope<Workgroup>, semantics = #spirv.memory_semantics<AcquireRelease>} : (!spirv.ptr<i32, StorageBuffer>, i64) -> (i64)
  return %0 : i64
}

// -----

func.func @atomic_and(%ptr : !spirv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // expected-error @+1 {{expected at most one of these four memory constraints to be set: `Acquire`, `Release`,`AcquireRelease` or `SequentiallyConsistent`}}
  %0 = spirv.AtomicAnd <Device> <Acquire|Release> %ptr, %value : !spirv.ptr<i32, StorageBuffer>
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.AtomicCompareExchange
//===----------------------------------------------------------------------===//

func.func @atomic_compare_exchange(%ptr: !spirv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 {
  // CHECK: spirv.AtomicCompareExchange <Workgroup> <Release> <Acquire> %{{.*}}, %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
  %0 = spirv.AtomicCompareExchange <Workgroup> <Release> <Acquire> %ptr, %value, %comparator: !spirv.ptr<i32, Workgroup>
  return %0: i32
}

// -----

func.func @atomic_compare_exchange(%ptr: !spirv.ptr<i32, Workgroup>, %value: i64, %comparator: i32) -> i32 {
  // expected-error @+1 {{'spirv.AtomicCompareExchange' op failed to verify that `value` type matches pointee type of `pointer`}}
  %0 = "spirv.AtomicCompareExchange"(%ptr, %value, %comparator) {memory_scope = #spirv.scope<Workgroup>, equal_semantics = #spirv.memory_semantics<AcquireRelease>, unequal_semantics = #spirv.memory_semantics<AcquireRelease>} : (!spirv.ptr<i32, Workgroup>, i64, i32) -> (i32)
  return %0: i32
}

// -----

func.func @atomic_compare_exchange(%ptr: !spirv.ptr<i32, Workgroup>, %value: i32, %comparator: i16) -> i32 {
  // expected-error @+1 {{'spirv.AtomicCompareExchange' op failed to verify that `comparator` type matches pointee type of `pointer`}}
  %0 = "spirv.AtomicCompareExchange"(%ptr, %value, %comparator) {memory_scope = #spirv.scope<Workgroup>, equal_semantics = #spirv.memory_semantics<AcquireRelease>, unequal_semantics = #spirv.memory_semantics<AcquireRelease>} : (!spirv.ptr<i32, Workgroup>, i32, i16) -> (i32)
  return %0: i32
}

// -----

func.func @atomic_compare_exchange(%ptr: !spirv.ptr<i64, Workgroup>, %value: i32, %comparator: i32) -> i32 {
  // expected-error @+1 {{spirv.AtomicCompareExchange' op failed to verify that `result` type matches pointee type of `pointer`}}
  %0 = "spirv.AtomicCompareExchange"(%ptr, %value, %comparator) {memory_scope = #spirv.scope<Workgroup>, equal_semantics = #spirv.memory_semantics<AcquireRelease>, unequal_semantics = #spirv.memory_semantics<AcquireRelease>} : (!spirv.ptr<i64, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.AtomicCompareExchangeWeak
//===----------------------------------------------------------------------===//

func.func @atomic_compare_exchange_weak(%ptr: !spirv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 {
  // CHECK: spirv.AtomicCompareExchangeWeak <Workgroup> <Release> <Acquire> %{{.*}}, %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
  %0 = spirv.AtomicCompareExchangeWeak <Workgroup> <Release> <Acquire> %ptr, %value, %comparator: !spirv.ptr<i32, Workgroup>
  return %0: i32
}

// -----

func.func @atomic_compare_exchange_weak(%ptr: !spirv.ptr<i32, Workgroup>, %value: i64, %comparator: i32) -> i32 {
  // expected-error @+1 {{'spirv.AtomicCompareExchangeWeak' op failed to verify that `value` type matches pointee type of `pointer`}}
  %0 = "spirv.AtomicCompareExchangeWeak"(%ptr, %value, %comparator) {memory_scope = #spirv.scope<Workgroup>, equal_semantics = #spirv.memory_semantics<AcquireRelease>, unequal_semantics = #spirv.memory_semantics<AcquireRelease>} : (!spirv.ptr<i32, Workgroup>, i64, i32) -> (i32)
  return %0: i32
}

// -----

func.func @atomic_compare_exchange_weak(%ptr: !spirv.ptr<i32, Workgroup>, %value: i32, %comparator: i16) -> i32 {
  // expected-error @+1 {{'spirv.AtomicCompareExchangeWeak' op failed to verify that `comparator` type matches pointee type of `pointer`}}
  %0 = "spirv.AtomicCompareExchangeWeak"(%ptr, %value, %comparator) {memory_scope = #spirv.scope<Workgroup>, equal_semantics = #spirv.memory_semantics<AcquireRelease>, unequal_semantics = #spirv.memory_semantics<AcquireRelease>} : (!spirv.ptr<i32, Workgroup>, i32, i16) -> (i32)
  return %0: i32
}

// -----

func.func @atomic_compare_exchange_weak(%ptr: !spirv.ptr<i64, Workgroup>, %value: i32, %comparator: i32) -> i32 {
  // expected-error @+1 {{'spirv.AtomicCompareExchangeWeak' op failed to verify that `result` type matches pointee type of `pointer`}}
  %0 = "spirv.AtomicCompareExchangeWeak"(%ptr, %value, %comparator) {memory_scope = #spirv.scope<Workgroup>, equal_semantics = #spirv.memory_semantics<AcquireRelease>, unequal_semantics = #spirv.memory_semantics<AcquireRelease>} : (!spirv.ptr<i64, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.AtomicExchange
//===----------------------------------------------------------------------===//

func.func @atomic_exchange(%ptr: !spirv.ptr<i32, Workgroup>, %value: i32) -> i32 {
  // CHECK: spirv.AtomicExchange <Workgroup> <Release> %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
  %0 = spirv.AtomicExchange <Workgroup> <Release> %ptr, %value: !spirv.ptr<i32, Workgroup>
  return %0: i32
}

// -----

func.func @atomic_exchange(%ptr: !spirv.ptr<i32, Workgroup>, %value: i64) -> i32 {
  // expected-error @+1 {{'spirv.AtomicExchange' op failed to verify that `value` type matches pointee type of `pointer`}}
  %0 = "spirv.AtomicExchange"(%ptr, %value) {memory_scope = #spirv.scope<Workgroup>, semantics = #spirv.memory_semantics<AcquireRelease>} : (!spirv.ptr<i32, Workgroup>, i64) -> (i32)
  return %0: i32
}

// -----

func.func @atomic_exchange(%ptr: !spirv.ptr<i64, Workgroup>, %value: i32) -> i32 {
  // expected-error @+1 {{'spirv.AtomicExchange' op failed to verify that `value` type matches pointee type of `pointer`}}
  %0 = "spirv.AtomicExchange"(%ptr, %value) {memory_scope = #spirv.scope<Workgroup>, semantics = #spirv.memory_semantics<AcquireRelease>} : (!spirv.ptr<i64, Workgroup>, i32) -> (i32)
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.AtomicIAdd
//===----------------------------------------------------------------------===//

func.func @atomic_iadd(%ptr : !spirv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spirv.AtomicIAdd <Workgroup> <None> %{{.*}}, %{{.*}} : !spirv.ptr<i32, StorageBuffer>
  %0 = spirv.AtomicIAdd <Workgroup> <None> %ptr, %value : !spirv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spirv.AtomicIDecrement
//===----------------------------------------------------------------------===//

func.func @atomic_idecrement(%ptr : !spirv.ptr<i32, StorageBuffer>) -> i32 {
  // CHECK: spirv.AtomicIDecrement <Workgroup> <None> %{{.*}} : !spirv.ptr<i32, StorageBuffer>
  %0 = spirv.AtomicIDecrement <Workgroup> <None> %ptr : !spirv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spirv.AtomicIIncrement
//===----------------------------------------------------------------------===//

func.func @atomic_iincrement(%ptr : !spirv.ptr<i32, StorageBuffer>) -> i32 {
  // CHECK: spirv.AtomicIIncrement <Workgroup> <None> %{{.*}} : !spirv.ptr<i32, StorageBuffer>
  %0 = spirv.AtomicIIncrement <Workgroup> <None> %ptr : !spirv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spirv.AtomicISub
//===----------------------------------------------------------------------===//

func.func @atomic_isub(%ptr : !spirv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spirv.AtomicISub <Workgroup> <None> %{{.*}}, %{{.*}} : !spirv.ptr<i32, StorageBuffer>
  %0 = spirv.AtomicISub <Workgroup> <None> %ptr, %value : !spirv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spirv.AtomicOr
//===----------------------------------------------------------------------===//

func.func @atomic_or(%ptr : !spirv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spirv.AtomicOr <Workgroup> <None> %{{.*}}, %{{.*}} : !spirv.ptr<i32, StorageBuffer>
  %0 = spirv.AtomicOr <Workgroup> <None> %ptr, %value : !spirv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spirv.AtomicSMax
//===----------------------------------------------------------------------===//

func.func @atomic_smax(%ptr : !spirv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spirv.AtomicSMax <Workgroup> <None> %{{.*}}, %{{.*}} : !spirv.ptr<i32, StorageBuffer>
  %0 = spirv.AtomicSMax <Workgroup> <None> %ptr, %value : !spirv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spirv.AtomicSMin
//===----------------------------------------------------------------------===//

func.func @atomic_smin(%ptr : !spirv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spirv.AtomicSMin <Workgroup> <None> %{{.*}}, %{{.*}} : !spirv.ptr<i32, StorageBuffer>
  %0 = spirv.AtomicSMin <Workgroup> <None> %ptr, %value : !spirv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spirv.AtomicUMax
//===----------------------------------------------------------------------===//

func.func @atomic_umax(%ptr : !spirv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spirv.AtomicUMax <Workgroup> <None> %{{.*}}, %{{.*}} : !spirv.ptr<i32, StorageBuffer>
  %0 = spirv.AtomicUMax <Workgroup> <None> %ptr, %value : !spirv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spirv.AtomicUMin
//===----------------------------------------------------------------------===//

func.func @atomic_umin(%ptr : !spirv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spirv.AtomicUMin <Workgroup> <None> %{{.*}}, %{{.*}} : !spirv.ptr<i32, StorageBuffer>
  %0 = spirv.AtomicUMin <Workgroup> <None> %ptr, %value : !spirv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spirv.AtomicXor
//===----------------------------------------------------------------------===//

func.func @atomic_xor(%ptr : !spirv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spirv.AtomicXor <Workgroup> <None> %{{.*}}, %{{.*}} : !spirv.ptr<i32, StorageBuffer>
  %0 = spirv.AtomicXor <Workgroup> <None> %ptr, %value : !spirv.ptr<i32, StorageBuffer>
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.EXT.AtomicFAdd
//===----------------------------------------------------------------------===//

func.func @atomic_fadd(%ptr : !spirv.ptr<f32, StorageBuffer>, %value : f32) -> f32 {
  // CHECK: spirv.EXT.AtomicFAdd <Device> <None> %{{.*}}, %{{.*}} : !spirv.ptr<f32, StorageBuffer>
  %0 = spirv.EXT.AtomicFAdd <Device> <None> %ptr, %value : !spirv.ptr<f32, StorageBuffer>
  return %0 : f32
}

// -----

func.func @atomic_fadd(%ptr : !spirv.ptr<i32, StorageBuffer>, %value : f32) -> f32 {
  // expected-error @+1 {{'spirv.EXT.AtomicFAdd' op failed to verify that `result` type matches pointee type of `pointer`}}
  %0 = "spirv.EXT.AtomicFAdd"(%ptr, %value) {memory_scope = #spirv.scope<Workgroup>, semantics = #spirv.memory_semantics<AcquireRelease>} : (!spirv.ptr<i32, StorageBuffer>, f32) -> (f32)
  return %0 : f32
}

// -----

func.func @atomic_fadd(%ptr : !spirv.ptr<f32, StorageBuffer>, %value : f64) -> f64 {
  // expected-error @+1 {{'spirv.EXT.AtomicFAdd' op failed to verify that `result` type matches pointee type of `pointer`}}
  %0 = "spirv.EXT.AtomicFAdd"(%ptr, %value) {memory_scope = #spirv.scope<Device>, semantics = #spirv.memory_semantics<AcquireRelease>} : (!spirv.ptr<f32, StorageBuffer>, f64) -> (f64)
  return %0 : f64
}

// -----

func.func @atomic_fadd(%ptr : !spirv.ptr<f32, StorageBuffer>, %value : f32) -> f32 {
  // expected-error @+1 {{expected at most one of these four memory constraints to be set: `Acquire`, `Release`,`AcquireRelease` or `SequentiallyConsistent`}}
  %0 = spirv.EXT.AtomicFAdd <Device> <Acquire|Release> %ptr, %value : !spirv.ptr<f32, StorageBuffer>
  return %0 : f32
}
