// RUN: mlir-opt --lower-host-to-llvm %s | FileCheck %s
  
module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_variable_pointers]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {

  //       CHECK: llvm.mlir.global linkonce @__spv__foo_bar_arg_0_descriptor_set0_binding0() : !llvm.struct<(array<6 x i32>)>
  //       CHECK: llvm.func @__spv__foo_bar()

  //       CHECK: spv.module @__spv__foo
  //       CHECK:   spv.GlobalVariable @bar_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<6 x i32, stride=4> [0])>, StorageBuffer>
  //       CHECK:   spv.func @__spv__foo_bar
  
  //       CHECK:   spv.EntryPoint "GLCompute" @__spv__foo_bar
  //       CHECK:   spv.ExecutionMode @__spv__foo_bar "LocalSize", 1, 1, 1

  // CHECK-LABEL: @main
  //       CHECK:   %[[SRC:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  //  CHECK-NEXT:   %[[DEST:.*]] = llvm.mlir.addressof @__spv__foo_bar_arg_0_descriptor_set0_binding0 : !llvm.ptr<struct<(array<6 x i32>)>>
  //  CHECK-NEXT:   llvm.mlir.constant(false) : i1
  //  CHECK-NEXT:   "llvm.intr.memcpy"(%[[DEST]], %[[SRC]], %[[SIZE:.*]], %{{.*}}) : (!llvm.ptr<struct<(array<6 x i32>)>>, !llvm.ptr<i32>, i64, i1) -> ()
  //  CHECK-NEXT:   llvm.call @__spv__foo_bar() : () -> ()
  //  CHECK-NEXT:   llvm.mlir.constant(false) : i1
  //  CHECK-NEXT:   "llvm.intr.memcpy"(%[[SRC]], %[[DEST]], %[[SIZE]], %{{.*}}) : (!llvm.ptr<i32>, !llvm.ptr<struct<(array<6 x i32>)>>, i64, i1) -> ()

  spv.module @__spv__foo Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_variable_pointers]> {
    spv.GlobalVariable @bar_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<6 x i32, stride=4> [0])>, StorageBuffer>
    spv.func @bar() "None" attributes {workgroup_attributions = 0 : i64} {
      %0 = spv.mlir.addressof @bar_arg_0 : !spv.ptr<!spv.struct<(!spv.array<6 x i32, stride=4> [0])>, StorageBuffer>
      spv.Return
    }
    spv.EntryPoint "GLCompute" @bar
    spv.ExecutionMode @bar "LocalSize", 1, 1, 1
  }

  gpu.module @foo {
    gpu.func @bar(%arg0: memref<6xi32>) kernel attributes {spv.entry_point_abi = {local_size = dense<1> : vector<3xi32>}} {
      gpu.return
    }
  }

  func @main() {
    %buffer = alloc() : memref<6xi32>
    %one = constant 1 : index
    gpu.launch_func @foo::@bar blocks in (%one, %one, %one)
        threads in (%one, %one, %one) args(%buffer : memref<6xi32>)
    return
  }
}
