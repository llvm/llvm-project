// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// `llvm.strictfp` marks the calls that named intrinsic ops translate to.

// CHECK-LABEL: define float @strictfp_intrinsics(
llvm.func @strictfp_intrinsics(%arg0: f32, %dst: !llvm.ptr, %src: !llvm.ptr, %len: i64) -> f32 attributes {strictfp} {
  // CHECK: call float @llvm.fabs.f32(float %{{.*}}) #[[STRICT:[0-9]+]]
  %0 = llvm.intr.fabs(%arg0) {llvm.strictfp} : (f32) -> f32
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i1 false) #[[STRICT]]
  "llvm.intr.memcpy"(%dst, %src, %len) <{isVolatile = false}> {llvm.strictfp} : (!llvm.ptr, !llvm.ptr, i64) -> ()
  // CHECK: call void @llvm.trap() #[[STRICT]]
  llvm.intr.trap {llvm.strictfp}
  // CHECK: call float @llvm.fabs.f32(float %{{.*}}){{$}}
  %1 = llvm.intr.fabs(%0) : (f32) -> f32
  llvm.return %1 : f32
}

// CHECK: attributes #[[STRICT]] = { strictfp }
