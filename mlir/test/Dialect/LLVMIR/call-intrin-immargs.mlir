// RUN: mlir-opt -verify-roundtrip %s | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode | mlir-opt | FileCheck %s
// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s --check-prefix=LLVM
// RUN: mlir-opt %s -canonicalize='region-simplify=aggressive' | mlir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

// CHECK-LABEL: llvm.func @round_ss
// CHECK: llvm.call_intrinsic "llvm.x86.sse41.round.ss"(%arg0, %arg1) {fastmathFlags = #llvm.fastmath<reassoc>, immarg_positions = array<i32: 2>, immarg_values = [1 : i32]}
// LLVM-LABEL: define <4 x float> @round_ss(
// LLVM: call reassoc <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 1)
llvm.func @round_ss(%a: vector<4xf32>, %b: vector<4xf32>) -> vector<4xf32> {
  %r = llvm.call_intrinsic "llvm.x86.sse41.round.ss"(%a, %b) {
    fastmathFlags = #llvm.fastmath<reassoc>,
    immarg_positions = array<i32: 2>, immarg_values = [1 : i32]
  } : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  llvm.return %r : vector<4xf32>
}

// Operand bundle inputs must not participate in overload resolution.
// CHECK-LABEL: llvm.func @ctlz_with_bundle
// CHECK: llvm.call_intrinsic "llvm.ctlz"(%arg0) ["tag"(%arg1 : i64)] {immarg_positions = array<i32: 1>, immarg_values = [false]} : (i32) -> i32
// LLVM-LABEL: define i32 @ctlz_with_bundle(
// LLVM: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false) [ "tag"(i64 %{{.*}}) ]
llvm.func @ctlz_with_bundle(%x: i32, %extra: i64) -> i32 {
  %r = llvm.call_intrinsic "llvm.ctlz"(%x) ["tag"(%extra : i64)] {
    immarg_positions = array<i32: 1>, immarg_values = [false]
  } : (i32) -> i32
  llvm.return %r : i32
}

// CHECK-LABEL: llvm.func @probability
// CHECK: llvm.call_intrinsic "llvm.expect.with.probability"(%arg0, %arg1) {immarg_positions = array<i32: 2>, immarg_values = [5.000000e-01]} : (i64, i64) -> i64
// LLVM-LABEL: define i64 @probability(
// LLVM: call i64 @llvm.expect.with.probability.i64(i64 %{{.*}}, i64 %{{.*}}, double 5.000000e-01)
llvm.func @probability(%x: i64, %expected: i64) -> i64 {
  %r = llvm.call_intrinsic "llvm.expect.with.probability"(%x, %expected) {
    immarg_positions = array<i32: 2>, immarg_values = [0.5 : f64]
  } : (i64, i64) -> i64
  llvm.return %r : i64
}

// CHECK-LABEL: llvm.func @prefetch
// CHECK: llvm.call_intrinsic "llvm.prefetch"(%arg0) {immarg_positions = array<i32: 1, 2, 3>, immarg_values = [0 : i32, 3 : i32, 1 : i32]} : (!llvm.ptr {llvm.nonnull}) -> ()
// LLVM-LABEL: define void @prefetch(
// LLVM: call void @llvm.prefetch.p0(ptr nonnull %{{.*}}, i32 0, i32 3, i32 1)
llvm.func @prefetch(%p: !llvm.ptr) {
  llvm.call_intrinsic "llvm.prefetch"(%p) {
    immarg_positions = array<i32: 1, 2, 3>,
    immarg_values = [0 : i32, 3 : i32, 1 : i32]
  } : (!llvm.ptr {llvm.nonnull}) -> ()
  llvm.return
}

// Argument attributes after an immediate must retain their LLVM argument index.
// CHECK-LABEL: llvm.func @invariant_end
// CHECK: llvm.call_intrinsic "llvm.invariant.end"(%arg0, %arg1) {immarg_positions = array<i32: 1>, immarg_values = [-1]} : (!llvm.ptr, !llvm.ptr {llvm.nonnull}) -> ()
// LLVM-LABEL: define void @invariant_end(
// LLVM: call void @llvm.invariant.end.p0(ptr %{{.*}}, i64 -1, ptr nonnull %{{.*}})
llvm.func @invariant_end(%start: !llvm.ptr, %p: !llvm.ptr) {
  llvm.call_intrinsic "llvm.invariant.end"(%start, %p) {
    immarg_positions = array<i32: 1>, immarg_values = [-1 : i64]
  } : (!llvm.ptr, !llvm.ptr {llvm.nonnull}) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @all_immediate
// CHECK: llvm.call_intrinsic "llvm.ubsantrap"() {immarg_positions = array<i32: 0>, immarg_values = [7 : i8]} : () -> ()
// LLVM-LABEL: define void @all_immediate(
// LLVM: call void @llvm.ubsantrap(i8 7)
llvm.func @all_immediate() {
  llvm.call_intrinsic "llvm.ubsantrap"() {
    immarg_positions = array<i32: 0>, immarg_values = [7 : i8]
  } : () -> ()
  llvm.unreachable
}

// Nonconsecutive immediates can have different types, and variadic arguments
// follow the fixed arguments in the reconstructed LLVM argument list.
// CHECK-LABEL: llvm.func @patchpoint
// CHECK: llvm.call_intrinsic "llvm.experimental.patchpoint.void"(%arg0, %arg1) {immarg_positions = array<i32: 0, 1, 3>, immarg_values = [42, 16 : i32, 0 : i32]} : (!llvm.ptr, i64) -> ()
// LLVM-LABEL: define void @patchpoint(
// LLVM: call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 42, i32 16, ptr %{{.*}}, i32 0, i64 %{{.*}})
llvm.func @patchpoint(%target: !llvm.ptr, %live: i64) {
  llvm.call_intrinsic "llvm.experimental.patchpoint.void"(%target, %live) {
    immarg_positions = array<i32: 0, 1, 3>,
    immarg_values = [42 : i64, 16 : i32, 0 : i32]
  } : (!llvm.ptr, i64) -> ()
  llvm.return
}

// Overloaded variadic intrinsics must resolve the declaration using only the
// fixed arguments, while preserving the variadic operands on the call.
// CHECK-LABEL: llvm.func @patchpoint_result
// CHECK: llvm.call_intrinsic "llvm.experimental.patchpoint"(%arg0, %arg1) {immarg_positions = array<i32: 0, 1, 3>, immarg_values = [42, 16 : i32, 0 : i32]} : (!llvm.ptr, i64) -> i64
// LLVM-LABEL: define i64 @patchpoint_result(
// LLVM: call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 42, i32 16, ptr %{{.*}}, i32 0, i64 %{{.*}})
llvm.func @patchpoint_result(%target: !llvm.ptr, %live: i64) -> i64 {
  %r = llvm.call_intrinsic "llvm.experimental.patchpoint"(%target, %live) {
    immarg_positions = array<i32: 0, 1, 3>,
    immarg_values = [42 : i64, 16 : i32, 0 : i32]
  } : (!llvm.ptr, i64) -> i64
  llvm.return %r : i64
}

// The declaration remains variadic even when there are no variadic arguments.
// CHECK-LABEL: llvm.func @patchpoint_no_varargs
// CHECK: llvm.call_intrinsic "llvm.experimental.patchpoint"(%arg0) {immarg_positions = array<i32: 0, 1, 3>, immarg_values = [43, 16 : i32, 0 : i32]} : (!llvm.ptr) -> i64
// LLVM-LABEL: define i64 @patchpoint_no_varargs(
// LLVM: call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 43, i32 16, ptr %{{.*}}, i32 0)
llvm.func @patchpoint_no_varargs(%target: !llvm.ptr) -> i64 {
  %r = llvm.call_intrinsic "llvm.experimental.patchpoint"(%target) {
    immarg_positions = array<i32: 0, 1, 3>,
    immarg_values = [43 : i64, 16 : i32, 0 : i32]
  } : (!llvm.ptr) -> i64
  llvm.return %r : i64
}

// Block merging must not turn different immediates into a block argument.
// CHECK-LABEL: llvm.func @different_modes
// CHECK: llvm.call_intrinsic "llvm.x86.sse41.round.ss"{{.*}}immarg_values = [1 : i32]
// CHECK: llvm.call_intrinsic "llvm.x86.sse41.round.ss"{{.*}}immarg_values = [2 : i32]
// LLVM-LABEL: define <4 x float> @different_modes(
// LLVM: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 1)
// LLVM: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 2)
llvm.func @different_modes(%cond: i1, %x: vector<4xf32>) -> vector<4xf32> {
  llvm.cond_br %cond, ^bb1, ^bb2
^bb1:
  %a = llvm.call_intrinsic "llvm.x86.sse41.round.ss"(%x, %x) {
    immarg_positions = array<i32: 2>, immarg_values = [1 : i32]
  } : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  llvm.return %a : vector<4xf32>
^bb2:
  %b = llvm.call_intrinsic "llvm.x86.sse41.round.ss"(%x, %x) {
    immarg_positions = array<i32: 2>, immarg_values = [2 : i32]
  } : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  llvm.return %b : vector<4xf32>
}
