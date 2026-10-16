; RUN: mlir-translate -import-llvm -prefer-unregistered-intrinsics %s | FileCheck %s --check-prefix=IMPORT
; RUN: mlir-translate -import-llvm -prefer-unregistered-intrinsics %s | mlir-opt -canonicalize='region-simplify=aggressive' | mlir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

declare i32 @llvm.ctlz.i32(i32, i1 immarg)
declare i64 @llvm.expect.with.probability.i64(i64, i64, double immarg)
declare void @llvm.prefetch.p0(ptr, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.invariant.end.p0(ptr, i64 immarg, ptr)
declare void @llvm.ubsantrap(i8 immarg)
declare void @llvm.experimental.patchpoint.void(i64 immarg, i32 immarg, ptr, i32 immarg, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64 immarg, i32 immarg, ptr, i32 immarg, ...)
declare <4 x float> @llvm.x86.sse41.round.ss(<4 x float>, <4 x float>, i32 immarg)

; The immarg attribute is inherited from the declaration. Bundle operands must
; remain SSA values, independently of the immediate argument positions.
; IMPORT-LABEL: llvm.func @ctlz_with_bundle
; IMPORT-NOT: llvm.mlir.constant
; IMPORT: llvm.call_intrinsic "llvm.ctlz.i32"(%arg0) ["tag"(%arg1 : i64)] {immarg_positions = array<i32: 1>, immarg_values = [false]} : (i32) -> i32
; LLVM-LABEL: define i32 @ctlz_with_bundle(
; LLVM: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false) [ "tag"(i64 %{{.*}}) ]
define i32 @ctlz_with_bundle(i32 %x, i64 %extra) {
  %r = call i32 @llvm.ctlz.i32(i32 %x, i1 false) ["tag"(i64 %extra)]
  ret i32 %r
}

; IMPORT-LABEL: llvm.func @probability
; IMPORT-NOT: llvm.mlir.constant
; IMPORT: llvm.call_intrinsic "llvm.expect.with.probability.i64"(%arg0, %arg1) {immarg_positions = array<i32: 2>, immarg_values = [5.000000e-01]} : (i64, i64) -> i64
; LLVM-LABEL: define i64 @probability(
; LLVM: call i64 @llvm.expect.with.probability.i64(i64 %{{.*}}, i64 %{{.*}}, double 5.000000e-01)
define i64 @probability(i64 %x, i64 %expected) {
  %r = call i64 @llvm.expect.with.probability.i64(i64 %x, i64 %expected, double 0.5)
  ret i64 %r
}

; IMPORT-LABEL: llvm.func @prefetch
; IMPORT-NOT: llvm.mlir.constant
; IMPORT: llvm.call_intrinsic "llvm.prefetch.p0"(%arg0) {immarg_positions = array<i32: 1, 2, 3>, immarg_values = [0 : i32, 3 : i32, 1 : i32]} : (!llvm.ptr {llvm.nonnull}) -> ()
; LLVM-LABEL: define void @prefetch(
; LLVM: call void @llvm.prefetch.p0(ptr nonnull %{{.*}}, i32 0, i32 3, i32 1)
define void @prefetch(ptr %p) {
  call void @llvm.prefetch.p0(ptr nonnull %p, i32 0, i32 3, i32 1)
  ret void
}

; IMPORT-LABEL: llvm.func @invariant_end
; IMPORT-NOT: llvm.mlir.constant
; IMPORT: llvm.call_intrinsic "llvm.invariant.end.p0"(%arg0, %arg1) {immarg_positions = array<i32: 1>, immarg_values = [-1]} : (!llvm.ptr, !llvm.ptr {llvm.nonnull}) -> ()
; LLVM-LABEL: define void @invariant_end(
; LLVM: call void @llvm.invariant.end.p0(ptr %{{.*}}, i64 -1, ptr nonnull %{{.*}})
define void @invariant_end(ptr %start, ptr %p) {
  call void @llvm.invariant.end.p0(ptr %start, i64 -1, ptr nonnull %p)
  ret void
}

; IMPORT-LABEL: llvm.func @all_immediate
; IMPORT-NOT: llvm.mlir.constant
; IMPORT: llvm.call_intrinsic "llvm.ubsantrap"() {immarg_positions = array<i32: 0>, immarg_values = [7 : i8]} : () -> ()
; LLVM-LABEL: define void @all_immediate(
; LLVM: call void @llvm.ubsantrap(i8 7)
define void @all_immediate() {
  call void @llvm.ubsantrap(i8 7)
  unreachable
}

; IMPORT-LABEL: llvm.func @patchpoint
; IMPORT-NOT: llvm.mlir.constant
; IMPORT: llvm.call_intrinsic "llvm.experimental.patchpoint.void"(%arg0, %arg1) {immarg_positions = array<i32: 0, 1, 3>, immarg_values = [42, 16 : i32, 0 : i32]} : (!llvm.ptr, i64) -> ()
; LLVM-LABEL: define void @patchpoint(
; LLVM: call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 42, i32 16, ptr %{{.*}}, i32 0, i64 %{{.*}})
define void @patchpoint(ptr %target, i64 %live) {
  call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 42, i32 16, ptr %target, i32 0, i64 %live)
  ret void
}

; Overloaded variadic intrinsic calls must retain both immediate and variadic
; arguments when translating back to LLVM IR.
; IMPORT-LABEL: llvm.func @patchpoint_result
; IMPORT-NOT: llvm.mlir.constant
; IMPORT: llvm.call_intrinsic "llvm.experimental.patchpoint.i64"(%arg0, %arg1) {immarg_positions = array<i32: 0, 1, 3>, immarg_values = [42, 16 : i32, 0 : i32]} : (!llvm.ptr, i64) -> i64
; LLVM-LABEL: define i64 @patchpoint_result(
; LLVM: call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 42, i32 16, ptr %{{.*}}, i32 0, i64 %{{.*}})
define i64 @patchpoint_result(ptr %target, i64 %live) {
  %r = call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 42, i32 16, ptr %target, i32 0, i64 %live)
  ret i64 %r
}

; Import must preserve the immediates before any MLIR optimization takes place.
; IMPORT-LABEL: llvm.func @different_modes
; IMPORT-NOT: llvm.mlir.constant
; IMPORT: llvm.call_intrinsic "llvm.x86.sse41.round.ss"{{.*}}immarg_values = [1 : i32]
; IMPORT: llvm.call_intrinsic "llvm.x86.sse41.round.ss"{{.*}}immarg_values = [2 : i32]
; LLVM-LABEL: define <4 x float> @different_modes(
; LLVM: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 1)
; LLVM: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 2)
define <4 x float> @different_modes(i1 %cond, <4 x float> %x) {
  br i1 %cond, label %bb1, label %bb2
bb1:
  %a = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %x, <4 x float> %x, i32 1)
  ret <4 x float> %a
bb2:
  %b = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %x, <4 x float> %x, i32 2)
  ret <4 x float> %b
}
