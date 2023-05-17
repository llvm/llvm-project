// RUN: mlir-translate -mlir-to-llvmir -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: ; ModuleID = 'LLVMDialectModule'
// CHECK: source_filename = "LLVMDialectModule"
// CHECK: declare ptr @malloc(i64)
// CHECK: declare void @free(ptr)
// CHECK: define <4 x float> @round_sse41() {
// CHECK:  %1 = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> <float 0x3FC99999A0000000, float 0x3FC99999A0000000, float 0x3FC99999A0000000, float 0x3FC99999A0000000>, <4 x float> <float 0x3FC99999A0000000, float 0x3FC99999A0000000, float 0x3FC99999A0000000, float 0x3FC99999A0000000>, i32 1)
// CHECK:  ret <4 x float> %1
// CHECK: }
llvm.func @round_sse41() -> vector<4xf32> {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(dense<0.2> : vector<4xf32>) : vector<4xf32>
    %res = llvm.call_intrinsic "llvm.x86.sse41.round.ss"(%1, %1, %0) : (vector<4xf32>, vector<4xf32>, i32) -> vector<4xf32> {}
    llvm.return %res: vector<4xf32>
}

// -----

// CHECK: ; ModuleID = 'LLVMDialectModule'
// CHECK: source_filename = "LLVMDialectModule"

// CHECK: declare ptr @malloc(i64)

// CHECK: declare void @free(ptr)

// CHECK: define float @round_overloaded() {
// CHECK:   %1 = call float @llvm.round.f32(float 1.000000e+00)
// CHECK:   ret float %1
// CHECK: }
llvm.func @round_overloaded() -> f32 {
    %0 = llvm.mlir.constant(1.0 : f32) : f32
    %res = llvm.call_intrinsic "llvm.round"(%0) : (f32) -> f32 {}
    llvm.return %res: f32
}

// -----

// CHECK: ; ModuleID = 'LLVMDialectModule'
// CHECK: source_filename = "LLVMDialectModule"
// CHECK: declare ptr @malloc(i64)
// CHECK: declare void @free(ptr)
// CHECK: define void @lifetime_start() {
// CHECK:   %1 = alloca float, i8 1, align 4
// CHECK:   call void @llvm.lifetime.start.p0(i64 4, ptr %1)
// CHECK:   ret void
// CHECK: }
llvm.func @lifetime_start() {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.mlir.constant(1 : i8) : i8
    %2 = llvm.alloca %1 x f32 : (i8) -> !llvm.ptr
    llvm.call_intrinsic "llvm.lifetime.start"(%0, %2) : (i64, !llvm.ptr) -> () {}
    llvm.return
}

// -----

llvm.func @variadic() {
    %0 = llvm.mlir.constant(1 : i8) : i8
    %1 = llvm.alloca %0 x f32 : (i8) -> !llvm.ptr
    llvm.call_intrinsic "llvm.localescape"(%1, %1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
}

// -----

llvm.func @no_intrinsic() {
    // expected-error@below {{'llvm.call_intrinsic' op couldn't find intrinsic: "llvm.does_not_exist"}}
    // expected-error@below {{LLVM Translation failed for operation: llvm.call_intrinsic}}
    llvm.call_intrinsic "llvm.does_not_exist"() : () -> ()
    llvm.return
}

// -----

llvm.func @bad_types() {
    %0 = llvm.mlir.constant(1 : i8) : i8
    // expected-error@below {{'llvm.call_intrinsic' op intrinsic type is not a match}}
    // expected-error@below {{LLVM Translation failed for operation: llvm.call_intrinsic}}
    llvm.call_intrinsic "llvm.round"(%0) : (i8) -> i8 {}
    llvm.return
}
