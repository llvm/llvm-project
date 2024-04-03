// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: @global = global target("spirv.DeviceEvent") zeroinitializer
llvm.mlir.global external @global() {addr_space = 0 : i32} : !llvm.target<"spirv.DeviceEvent"> {
  %0 = llvm.mlir.zero : !llvm.target<"spirv.DeviceEvent">
  llvm.return %0 : !llvm.target<"spirv.DeviceEvent">
}

// CHECK-LABEL: define target("spirv.Event") @func2() {
// CHECK-NEXT:    %1 = alloca target("spirv.Event"), align 8
// CHECK-NEXT:    %2 = load target("spirv.Event"), ptr %1, align 8
// CHECK-NEXT:    ret target("spirv.Event") poison
llvm.func @func2() -> !llvm.target<"spirv.Event"> {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.poison : !llvm.target<"spirv.Event">
  %2 = llvm.alloca %0 x !llvm.target<"spirv.Event"> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = ptr.load %2 {alignment = 8 : i64} : !llvm.ptr -> !llvm.target<"spirv.Event">
  llvm.return %1 : !llvm.target<"spirv.Event">
}

// CHECK-LABEL: define void @func3() {
// CHECK-NEXT:    %1 = freeze target("spirv.DeviceEvent") zeroinitializer
// CHECK-NEXT:    ret void
llvm.func @func3() {
  %0 = llvm.mlir.zero : !llvm.target<"spirv.DeviceEvent">
  %1 = llvm.freeze %0 : !llvm.target<"spirv.DeviceEvent">
  llvm.return
}
