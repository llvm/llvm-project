// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s
// RUN: mlir-opt -convert-spirv-to-llvm='allow-pattern-rollback=0' %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Read-modify-write ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @atomic_rmw
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr, %[[VAL:.*]]: i32)
spirv.func @atomic_rmw(%ptr: !spirv.ptr<i32, CrossWorkgroup>, %value: i32) "None" {
  // CHECK: %[[OLD:.*]] = llvm.atomicrmw _and %[[PTR]], %[[VAL]] syncscope("device") monotonic : !llvm.ptr, i32
  %0 = spirv.AtomicAnd <Device> <None> %ptr, %value : !spirv.ptr<i32, CrossWorkgroup>
  // CHECK: llvm.atomicrmw xchg %[[PTR]], %[[OLD]] syncscope("workgroup") acquire : !llvm.ptr, i32
  %1 = spirv.AtomicExchange <Workgroup> <Acquire> %ptr, %0 : !spirv.ptr<i32, CrossWorkgroup>
  // CHECK: llvm.atomicrmw add %[[PTR]], %[[VAL]] syncscope("subgroup") release : !llvm.ptr, i32
  %2 = spirv.AtomicIAdd <Subgroup> <Release> %ptr, %value : !spirv.ptr<i32, CrossWorkgroup>
  // CHECK: llvm.atomicrmw sub %[[PTR]], %[[VAL]] syncscope("singlethread") acq_rel : !llvm.ptr, i32
  %3 = spirv.AtomicISub <Invocation> <AcquireRelease> %ptr, %value : !spirv.ptr<i32, CrossWorkgroup>
  // CHECK: llvm.atomicrmw _or %[[PTR]], %[[VAL]] seq_cst : !llvm.ptr, i32
  %4 = spirv.AtomicOr <CrossDevice> <SequentiallyConsistent> %ptr, %value : !spirv.ptr<i32, CrossWorkgroup>
  // CHECK: llvm.atomicrmw volatile _xor %[[PTR]], %[[VAL]] syncscope("device") acq_rel : !llvm.ptr, i32
  %5 = spirv.AtomicXor <Device> <AcquireRelease|CrossWorkgroupMemory|Volatile> %ptr, %value : !spirv.ptr<i32, CrossWorkgroup>
  // CHECK: llvm.atomicrmw max %[[PTR]], %[[VAL]] syncscope("device") monotonic : !llvm.ptr, i32
  %6 = spirv.AtomicSMax <Device> <None> %ptr, %value : !spirv.ptr<i32, CrossWorkgroup>
  // CHECK: llvm.atomicrmw min %[[PTR]], %[[VAL]] syncscope("device") monotonic : !llvm.ptr, i32
  %7 = spirv.AtomicSMin <Device> <None> %ptr, %value : !spirv.ptr<i32, CrossWorkgroup>
  // CHECK: llvm.atomicrmw umax %[[PTR]], %[[VAL]] syncscope("device") monotonic : !llvm.ptr, i32
  %8 = spirv.AtomicUMax <Device> <None> %ptr, %value : !spirv.ptr<i32, CrossWorkgroup>
  // CHECK: llvm.atomicrmw umin %[[PTR]], %[[VAL]] syncscope("device") monotonic : !llvm.ptr, i32
  %9 = spirv.AtomicUMin <Device> <None> %ptr, %value : !spirv.ptr<i32, CrossWorkgroup>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.atomicrmw add %[[PTR]], %[[ONE]] syncscope("device") monotonic : !llvm.ptr, i32
  %10 = spirv.AtomicIIncrement <Device> <None> %ptr : !spirv.ptr<i32, CrossWorkgroup>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.atomicrmw sub %[[PTR]], %[[ONE]] syncscope("device") monotonic : !llvm.ptr, i32
  %11 = spirv.AtomicIDecrement <Device> <None> %ptr : !spirv.ptr<i32, CrossWorkgroup>
  spirv.Return
}

// CHECK-LABEL: @atomic_float
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr, %[[VAL:.*]]: f32)
spirv.func @atomic_float(%ptr: !spirv.ptr<f32, CrossWorkgroup>, %value: f32) "None" {
  // CHECK: llvm.atomicrmw fadd %[[PTR]], %[[VAL]] syncscope("device") monotonic : !llvm.ptr, f32
  %0 = spirv.EXT.AtomicFAdd <Device> <None> %ptr, %value : !spirv.ptr<f32, CrossWorkgroup>
  // CHECK: llvm.atomicrmw xchg %[[PTR]], %[[VAL]] syncscope("device") monotonic : !llvm.ptr, f32
  %1 = spirv.AtomicExchange <Device> <None> %ptr, %value : !spirv.ptr<f32, CrossWorkgroup>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.AtomicCompareExchange and spirv.AtomicCompareExchangeWeak
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @atomic_compare_exchange
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr, %[[VAL:.*]]: i64, %[[CMP:.*]]: i64)
spirv.func @atomic_compare_exchange(%ptr: !spirv.ptr<i64, CrossWorkgroup>, %value: i64, %comparator: i64) "None" {
  // CHECK: %[[PAIR:.*]] = llvm.cmpxchg %[[PTR]], %[[CMP]], %[[VAL]] syncscope("workgroup") acq_rel acquire : !llvm.ptr, i64
  // CHECK: %[[OLD:.*]] = llvm.extractvalue %[[PAIR]][0] : !llvm.struct<(i64, i1)>
  %0 = spirv.AtomicCompareExchange <Workgroup> <AcquireRelease> <Acquire> %ptr, %value, %comparator : !spirv.ptr<i64, CrossWorkgroup>
  // CHECK: %[[PAIR:.*]] = llvm.cmpxchg volatile %[[PTR]], %[[CMP]], %[[OLD]] syncscope("device") seq_cst monotonic : !llvm.ptr, i64
  // CHECK: llvm.extractvalue %[[PAIR]][0] : !llvm.struct<(i64, i1)>
  %1 = spirv.AtomicCompareExchangeWeak <Device> <SequentiallyConsistent> <Volatile> %ptr, %0, %comparator : !spirv.ptr<i64, CrossWorkgroup>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.AtomicLoad and spirv.AtomicStore
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @atomic_load_store
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr, %[[VAL:.*]]: i64, %[[FPTR:.*]]: !llvm.ptr, %[[FVAL:.*]]: f16)
spirv.func @atomic_load_store(%ptr: !spirv.ptr<i64, CrossWorkgroup>, %value: i64, %fptr: !spirv.ptr<f16, Workgroup>, %fvalue: f16) "None" {
  // CHECK: %[[LOADED:.*]] = llvm.load %[[PTR]] atomic syncscope("device") acquire <alignment = 8> : !llvm.ptr -> i64
  %0 = spirv.AtomicLoad <Device> <Acquire> %ptr : !spirv.ptr<i64, CrossWorkgroup>
  // CHECK: llvm.store volatile %[[LOADED]], %[[PTR]] atomic syncscope("device") release <alignment = 8> : i64, !llvm.ptr
  spirv.AtomicStore <Device> <Release|Volatile> %ptr, %0 : !spirv.ptr<i64, CrossWorkgroup>
  // CHECK: llvm.load %[[FPTR]] atomic syncscope("workgroup") seq_cst <alignment = 2> : !llvm.ptr -> f16
  %1 = spirv.AtomicLoad <Workgroup> <SequentiallyConsistent> %fptr : !spirv.ptr<f16, Workgroup>
  // CHECK: llvm.store %[[FVAL]], %[[FPTR]] atomic syncscope("workgroup") monotonic <alignment = 2> : f16, !llvm.ptr
  spirv.AtomicStore <Workgroup> <None> %fptr, %fvalue : !spirv.ptr<f16, Workgroup>
  // CHECK: llvm.load %[[PTR]] atomic syncscope("device") monotonic <alignment = 8> : !llvm.ptr -> i64
  %2 = spirv.AtomicLoad <Device> <None> %ptr : !spirv.ptr<i64, CrossWorkgroup>
  // CHECK: llvm.store %[[VAL]], %[[PTR]] atomic seq_cst <alignment = 8> : i64, !llvm.ptr
  spirv.AtomicStore <CrossDevice> <SequentiallyConsistent> %ptr, %value : !spirv.ptr<i64, CrossWorkgroup>
  spirv.Return
}
