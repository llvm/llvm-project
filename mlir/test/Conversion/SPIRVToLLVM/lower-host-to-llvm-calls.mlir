// RUN: mlir-opt --lower-host-to-llvm='use-opaque-pointers=1' %s -split-input-file | FileCheck %s

module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_variable_pointers]>, #spirv.resource_limits<max_compute_workgroup_invocations = 128, max_compute_workgroup_size = [128, 128, 64]>>} {

  //       CHECK: llvm.mlir.global linkonce @__spv__foo_bar_arg_0_descriptor_set0_binding0() {addr_space = 0 : i32} : !llvm.struct<(array<6 x i32>)>
  //       CHECK: llvm.func @__spv__foo_bar()

  //       CHECK: spirv.module @__spv__foo
  //       CHECK:   spirv.GlobalVariable @bar_arg_0 bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.array<6 x i32, stride=4> [0])>, StorageBuffer>
  //       CHECK:   spirv.func @__spv__foo_bar

  //       CHECK:   spirv.EntryPoint "GLCompute" @__spv__foo_bar
  //       CHECK:   spirv.ExecutionMode @__spv__foo_bar "LocalSize", 1, 1, 1

  // CHECK-LABEL: @main
  //       CHECK:   %[[SRC:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  //  CHECK-NEXT:   %[[DEST:.*]] = llvm.mlir.addressof @__spv__foo_bar_arg_0_descriptor_set0_binding0 : !llvm.ptr
  //  CHECK-NEXT:   "llvm.intr.memcpy"(%[[DEST]], %[[SRC]], %[[SIZE:.*]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  //  CHECK-NEXT:   llvm.call @__spv__foo_bar() : () -> ()
  //  CHECK-NEXT:   "llvm.intr.memcpy"(%[[SRC]], %[[DEST]], %[[SIZE]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()

  spirv.module @__spv__foo Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_variable_pointers]> {
    spirv.GlobalVariable @bar_arg_0 bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.array<6 x i32, stride=4> [0])>, StorageBuffer>
    spirv.func @bar() "None" attributes {workgroup_attributions = 0 : i64} {
      %0 = spirv.mlir.addressof @bar_arg_0 : !spirv.ptr<!spirv.struct<(!spirv.array<6 x i32, stride=4> [0])>, StorageBuffer>
      spirv.Return
    }
    spirv.EntryPoint "GLCompute" @bar
    spirv.ExecutionMode @bar "LocalSize", 1, 1, 1
  }

  gpu.module @foo {
    gpu.func @bar(%arg0: memref<6xi32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      gpu.return
    }
  }

  func.func @main() {
    %buffer = memref.alloc() : memref<6xi32>
    %one = arith.constant 1 : index
    gpu.launch_func @foo::@bar blocks in (%one, %one, %one)
        threads in (%one, %one, %one) args(%buffer : memref<6xi32>)
    return
  }
}

// -----

// Check using a specified sym_name attribute.
module {
  spirv.module Logical GLSL450 attributes { sym_name = "spirv.sym" } {
    // CHECK: spirv.func @spirv.sym_bar
    // CHECK: spirv.EntryPoint "GLCompute" @spirv.sym_bar
    // CHECK: spirv.ExecutionMode @spirv.sym_bar "LocalSize", 1, 1, 1
    spirv.func @bar() "None" {
      spirv.Return
    }
    spirv.EntryPoint "GLCompute" @bar
    spirv.ExecutionMode @bar "LocalSize", 1, 1, 1
  }
}

// -----

// Check using the default sym_name attribute.
module {
  spirv.module Logical GLSL450 {
    // CHECK: spirv.func @__spv___bar
    // CHECK: spirv.EntryPoint "GLCompute" @__spv___bar
    // CHECK: spirv.ExecutionMode @__spv___bar "LocalSize", 1, 1, 1
    spirv.func @bar() "None" {
      spirv.Return
    }
    spirv.EntryPoint "GLCompute" @bar
    spirv.ExecutionMode @bar "LocalSize", 1, 1, 1
  }
}
