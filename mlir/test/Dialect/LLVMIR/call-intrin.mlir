// RUN: mlir-translate -mlir-to-llvmir -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: define <4 x float> @round_sse41() {
// CHECK:  %1 = call reassoc <4 x float> @llvm.x86.sse41.round.ss(<4 x float> {{.*}}, <4 x float> {{.*}}, i32 1)
// CHECK:  ret <4 x float> %1
// CHECK: }
llvm.func @round_sse41() -> vector<4xf32> {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(dense<0.2> : vector<4xf32>) : vector<4xf32>
  %res = llvm.call_intrinsic "llvm.x86.sse41.round.ss"(%1, %1, %0) {fastmathFlags = #llvm.fastmath<reassoc>} : (vector<4xf32>, vector<4xf32>, i32) -> vector<4xf32>
  llvm.return %res: vector<4xf32>
}

// -----

// CHECK: define float @round_overloaded() {
// CHECK:   %1 = call float @llvm.round.f32(float 1.000000e+00)
// CHECK:   ret float %1
// CHECK: }
llvm.func @round_overloaded() -> f32 {
  %0 = llvm.mlir.constant(1.0 : f32) : f32
  %res = llvm.call_intrinsic "llvm.round"(%0) {} : (f32) -> f32
  llvm.return %res: f32
}

// -----

// CHECK: define void @lifetime_start() {
// CHECK:   %1 = alloca float, i8 1, align 4
// CHECK:   call void @llvm.lifetime.start.p0(ptr %1)
// CHECK:   ret void
// CHECK: }
llvm.func @lifetime_start() {
  %0 = llvm.mlir.constant(1 : i8) : i8
  %1 = llvm.alloca %0 x f32 : (i8) -> !llvm.ptr
  llvm.call_intrinsic "llvm.lifetime.start"(%1) {} : (!llvm.ptr) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: define void @variadic()
llvm.func @variadic() {
  %0 = llvm.mlir.constant(1 : i8) : i8
  %1 = llvm.alloca %0 x f32 : (i8) -> !llvm.ptr
  // CHECK: call void (...) @llvm.localescape(ptr %1, ptr %1)
  llvm.call_intrinsic "llvm.localescape"(%1, %1) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----

llvm.func @no_intrinsic() {
  // expected-error@below {{could not find LLVM intrinsic: "llvm.does_not_exist"}}
  // expected-error@below {{LLVM Translation failed for operation: llvm.call_intrinsic}}
  llvm.call_intrinsic "llvm.does_not_exist"() : () -> ()
  llvm.return
}

// -----

llvm.func @bad_types() {
  %0 = llvm.mlir.constant(1 : i8) : i8
  // expected-error@below {{call intrinsic signature i8 (i8) to overloaded intrinsic "llvm.round" does not match any of the overloads}}
  // expected-error@below {{LLVM Translation failed for operation: llvm.call_intrinsic}}
  llvm.call_intrinsic "llvm.round"(%0) {} : (i8) -> i8
  llvm.return
}

// -----

llvm.func @bad_result() {
  // expected-error @below {{intrinsic call returns void but "llvm.x86.sse41.round.ss" actually returns <4 x float>}}
  // expected-error@below {{LLVM Translation failed for operation: llvm.call_intrinsic}}
  llvm.call_intrinsic "llvm.x86.sse41.round.ss"() : () -> ()
  llvm.return
}

// -----

llvm.func @bad_result() {
  // expected-error @below {{intrinsic call returns <8 x float> but "llvm.x86.sse41.round.ss" actually returns <4 x float>}}
  // expected-error@below {{LLVM Translation failed for operation: llvm.call_intrinsic}}
  llvm.call_intrinsic "llvm.x86.sse41.round.ss"() : () -> (vector<8xf32>)
  llvm.return
}

// -----

llvm.func @bad_args() {
  // expected-error @below {{intrinsic call has 0 operands but "llvm.x86.sse41.round.ss" expects 3}}
  // expected-error@below {{LLVM Translation failed for operation: llvm.call_intrinsic}}
  llvm.call_intrinsic "llvm.x86.sse41.round.ss"() : () -> (vector<4xf32>)
  llvm.return
}

// -----

llvm.func @bad_args() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.mlir.constant(dense<0.2> : vector<4xf32>) : vector<4xf32>
  // expected-error @below {{intrinsic call operand #2 has type i64 but "llvm.x86.sse41.round.ss" expects i32}}
  // expected-error@below {{LLVM Translation failed for operation: llvm.call_intrinsic}}
  %res = llvm.call_intrinsic "llvm.x86.sse41.round.ss"(%1, %1, %0) {fastmathFlags = #llvm.fastmath<reassoc>} : (vector<4xf32>, vector<4xf32>, i64) -> vector<4xf32>
  llvm.return
}

// -----

// CHECK-LABEL: intrinsic_call_arg_attrs
llvm.func @intrinsic_call_arg_attrs(%arg0: i32) -> i32 {
  // CHECK: call i32 @llvm.riscv.sha256sig0(i32 signext %{{.*}})
  %0 = llvm.call_intrinsic "llvm.riscv.sha256sig0"(%arg0) : (i32 {llvm.signext}) -> (i32)
  llvm.return %0 : i32
}

// -----

// CHECK-LABEL: intrinsic_element_type
llvm.func @intrinsic_element_type(%arg0: !llvm.ptr) {
  // CHECK: call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i8) %{{.*}})
  %0 = llvm.call_intrinsic "llvm.aarch64.ldxr.p0"(%arg0) : (!llvm.ptr {llvm.elementtype = i8}) -> i64
  llvm.return
}
