// RUN: mlir-opt %s -launch-func-to-vulkan | FileCheck %s

// CHECK: llvm.mlir.global internal constant @kernel_spv_entry_point_name
// CHECK: llvm.mlir.global internal constant @SPIRV_BIN
// CHECK: %[[Vulkan_Runtime_ptr:.*]] = llvm.call @initVulkan() : () -> !llvm.ptr<i8>
// CHECK: %[[addressof_SPIRV_BIN:.*]] = llvm.mlir.addressof @SPIRV_BIN
// CHECK: %[[SPIRV_BIN_ptr:.*]] = llvm.getelementptr %[[addressof_SPIRV_BIN]]
// CHECK: %[[SPIRV_BIN_size:.*]] = llvm.mlir.constant
// CHECK: llvm.call @bindMemRef1DFloat(%[[Vulkan_Runtime_ptr]], %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, i32, i32, !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>) -> ()
// CHECK: llvm.call @setBinaryShader(%[[Vulkan_Runtime_ptr]], %[[SPIRV_BIN_ptr]], %[[SPIRV_BIN_size]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i32) -> ()
// CHECK: %[[addressof_entry_point:.*]] = llvm.mlir.addressof @kernel_spv_entry_point_name
// CHECK: %[[entry_point_ptr:.*]] = llvm.getelementptr %[[addressof_entry_point]]
// CHECK: llvm.call @setEntryPoint(%[[Vulkan_Runtime_ptr]], %[[entry_point_ptr]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
// CHECK: llvm.call @setNumWorkGroups(%[[Vulkan_Runtime_ptr]], %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, i64, i64, i64) -> ()
// CHECK: llvm.call @runOnVulkan(%[[Vulkan_Runtime_ptr]]) : (!llvm.ptr<i8>) -> ()
// CHECK: llvm.call @deinitVulkan(%[[Vulkan_Runtime_ptr]]) : (!llvm.ptr<i8>) -> ()

// CHECK: llvm.func @bindMemRef1DHalf(!llvm.ptr<i8>, i32, i32, !llvm.ptr<struct<(ptr<i16>, ptr<i16>, i64, array<1 x i64>, array<1 x i64>)>>)

module attributes {gpu.container_module} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @foo() {
    %0 = llvm.mlir.constant(12 : index) : i64
    %1 = llvm.mlir.null : !llvm.ptr<f32>
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.getelementptr %1[%2] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %4 = llvm.ptrtoint %3 : !llvm.ptr<f32> to i64
    %5 = llvm.mul %0, %4 : i64
    %6 = llvm.call @malloc(%5) : (i64) -> !llvm.ptr<i8>
    %7 = llvm.bitcast %6 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %7, %9[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = llvm.insertvalue %11, %10[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.insertvalue %0, %12[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %13, %14[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.extractvalue %15[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.extractvalue %15[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @vulkanLaunch(%16, %16, %16, %17, %18, %19, %20, %21) {spirv_blob = "\03\02#\07\00", spirv_entry_point = "kernel"}
    : (i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64) -> ()
    llvm.return
  }
  llvm.func @vulkanLaunch(%arg0: i64, %arg1: i64, %arg2: i64, %arg6: !llvm.ptr<f32>, %arg7: !llvm.ptr<f32>, %arg8: i64, %arg9: i64, %arg10: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg6, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg7, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg8, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg9, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg10, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %5, %7 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.call @_mlir_ciface_vulkanLaunch(%arg0, %arg1, %arg2, %7) : (i64, i64, i64, !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_vulkanLaunch(i64, i64, i64, !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>)
}
