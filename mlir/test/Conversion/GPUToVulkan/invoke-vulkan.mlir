// RUN: mlir-opt %s -launch-func-to-vulkan | FileCheck %s

// CHECK: llvm.mlir.global internal constant @kernel_spv_entry_point_name
// CHECK: llvm.mlir.global internal constant @SPIRV_BIN
// CHECK: %[[Vulkan_Runtime_ptr:.*]] = llvm.call @initVulkan() : () -> !llvm<"i8*">
// CHECK: %[[addressof_SPIRV_BIN:.*]] = llvm.mlir.addressof @SPIRV_BIN
// CHECK: %[[SPIRV_BIN_ptr:.*]] = llvm.getelementptr %[[addressof_SPIRV_BIN]]
// CHECK: %[[SPIRV_BIN_size:.*]] = llvm.mlir.constant
// CHECK: llvm.call @bindMemRef1DFloat(%[[Vulkan_Runtime_ptr]], %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm<"i8*">, !llvm.i32, !llvm.i32, !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">) -> !llvm.void
// CHECK: llvm.call @setBinaryShader(%[[Vulkan_Runtime_ptr]], %[[SPIRV_BIN_ptr]], %[[SPIRV_BIN_size]]) : (!llvm<"i8*">, !llvm<"i8*">, !llvm.i32) -> !llvm.void
// CHECK: %[[addressof_entry_point:.*]] = llvm.mlir.addressof @kernel_spv_entry_point_name
// CHECK: %[[entry_point_ptr:.*]] = llvm.getelementptr %[[addressof_entry_point]]
// CHECK: llvm.call @setEntryPoint(%[[Vulkan_Runtime_ptr]], %[[entry_point_ptr]]) : (!llvm<"i8*">, !llvm<"i8*">) -> !llvm.void
// CHECK: llvm.call @setNumWorkGroups(%[[Vulkan_Runtime_ptr]], %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm<"i8*">, !llvm.i64, !llvm.i64, !llvm.i64) -> !llvm.void
// CHECK: llvm.call @runOnVulkan(%[[Vulkan_Runtime_ptr]]) : (!llvm<"i8*">) -> !llvm.void
// CHECK: llvm.call @deinitVulkan(%[[Vulkan_Runtime_ptr]]) : (!llvm<"i8*">) -> !llvm.void

module attributes {gpu.container_module} {
  llvm.func @malloc(!llvm.i64) -> !llvm<"i8*">
  llvm.func @foo() {
    %0 = llvm.mlir.constant(12 : index) : !llvm.i64
    %1 = llvm.mlir.null : !llvm<"float*">
    %2 = llvm.mlir.constant(1 : index) : !llvm.i64
    %3 = llvm.getelementptr %1[%2] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
    %4 = llvm.ptrtoint %3 : !llvm<"float*"> to !llvm.i64
    %5 = llvm.mul %0, %4 : !llvm.i64
    %6 = llvm.call @malloc(%5) : (!llvm.i64) -> !llvm<"i8*">
    %7 = llvm.bitcast %6 : !llvm<"i8*"> to !llvm<"float*">
    %8 = llvm.mlir.undef : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %9 = llvm.insertvalue %7, %8[0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %10 = llvm.insertvalue %7, %9[1] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %11 = llvm.mlir.constant(0 : index) : !llvm.i64
    %12 = llvm.insertvalue %11, %10[2] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %13 = llvm.mlir.constant(1 : index) : !llvm.i64
    %14 = llvm.insertvalue %0, %12[3, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %15 = llvm.insertvalue %13, %14[4, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %16 = llvm.mlir.constant(1 : index) : !llvm.i64
    %17 = llvm.extractvalue %15[0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %18 = llvm.extractvalue %15[1] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %19 = llvm.extractvalue %15[2] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %20 = llvm.extractvalue %15[3, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %21 = llvm.extractvalue %15[4, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    llvm.call @vulkanLaunch(%16, %16, %16, %16, %16, %16, %17, %18, %19, %20, %21) {spirv_blob = "\03\02#\07\00", spirv_entry_point = "kernel"}
    : (!llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm<"float*">, !llvm<"float*">, !llvm.i64, !llvm.i64, !llvm.i64) -> ()
    llvm.return
  }
  llvm.func @vulkanLaunch(%arg0: !llvm.i64, %arg1: !llvm.i64, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm<"float*">, %arg7: !llvm<"float*">, %arg8: !llvm.i64, %arg9: !llvm.i64, %arg10: !llvm.i64) {
    %0 = llvm.mlir.undef : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %1 = llvm.insertvalue %arg6, %0[0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %2 = llvm.insertvalue %arg7, %1[1] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %3 = llvm.insertvalue %arg8, %2[2] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %4 = llvm.insertvalue %arg9, %3[3, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %5 = llvm.insertvalue %arg10, %4[4, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    %6 = llvm.mlir.constant(1 : index) : !llvm.i64
    %7 = llvm.alloca %6 x !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }"> : (!llvm.i64) -> !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">
    llvm.store %5, %7 : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">
    llvm.call @_mlir_ciface_vulkanLaunch(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %7) : (!llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_vulkanLaunch(!llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">)
}
