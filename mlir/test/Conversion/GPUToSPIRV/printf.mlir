// RUN: mlir-opt -allow-unregistered-dialect -split-input-file  -convert-gpu-to-spirv -verify-diagnostics %s | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int8, Kernel], []>, #spirv.resource_limits<>>
} {
  func.func @main() {
    %c1 = arith.constant 1 : index

    gpu.launch_func @kernels::@printf
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args()
    return
  }
  
  gpu.module @kernels {
    // CHECK: spirv.module @{{.*}} Physical32 OpenCL
    // CHECK-DAG: spirv.SpecConstant [[SPECCST:@.*]] = {{.*}} : i8
    // CHECK-DAG: spirv.SpecConstantComposite [[SPECCSTCOMPOSITE:@.*]] ([[SPECCST]], {{.*}}) : !spirv.array<[[ARRAYSIZE:.*]] x i8>
    // CHECK-DAG: spirv.GlobalVariable [[PRINTMSG:@.*]] initializer([[SPECCSTCOMPOSITE]]) {Constant}  : !spirv.ptr<!spirv.array<[[ARRAYSIZE]] x i8>, UniformConstant>
    gpu.func @printf() kernel
      attributes 
        {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
          // CHECK: [[FMTSTR_ADDR:%.*]] = spirv.mlir.addressof [[PRINTMSG]] : !spirv.ptr<!spirv.array<[[ARRAYSIZE]] x i8>, UniformConstant>
          // CHECK-NEXT: [[FMTSTR_PTR:%.*]] = spirv.Bitcast [[FMTSTR_ADDR]] : !spirv.ptr<!spirv.array<[[ARRAYSIZE]] x i8>, UniformConstant> to !spirv.ptr<i8, UniformConstant>
          // CHECK-NEXT {{%.*}} = spirv.CL.printf [[FMTSTR_PTR]] : !spirv.ptr<i8, UniformConstant> -> i32
          gpu.printf "\nHello\n"
          // CHECK: spirv.Return
          gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int8, Kernel], []>, #spirv.resource_limits<>>
} {
  func.func @main() {
    %c1   = arith.constant 1 : index
    %c100 = arith.constant 100: i32
    %cst_f32 = arith.constant 314.4: f32

    gpu.launch_func @kernels1::@printf_args
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%c100: i32, %cst_f32: f32)
    return
  }

   gpu.module @kernels1 {
    // CHECK: spirv.module @{{.*}} Physical32 OpenCL {
    // CHECK-DAG: spirv.SpecConstant [[SPECCST:@.*]] = {{.*}} : i8
    // CHECK-DAG: spirv.SpecConstantComposite [[SPECCSTCOMPOSITE:@.*]] ([[SPECCST]], {{.*}}) : !spirv.array<[[ARRAYSIZE:.*]] x i8>
    // CHECK-DAG: spirv.GlobalVariable [[PRINTMSG:@.*]] initializer([[SPECCSTCOMPOSITE]]) {Constant}  : !spirv.ptr<!spirv.array<[[ARRAYSIZE]] x i8>, UniformConstant>
    gpu.func @printf_args(%arg0: i32, %arg1: f32) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
        %0 = gpu.block_id x
        %1 = gpu.block_id y
        %2 = gpu.thread_id x

        // CHECK: [[FMTSTR_ADDR:%.*]] = spirv.mlir.addressof [[PRINTMSG]] : !spirv.ptr<!spirv.array<[[ARRAYSIZE]] x i8>, UniformConstant>
        // CHECK-NEXT: [[FMTSTR_PTR1:%.*]] = spirv.Bitcast [[FMTSTR_ADDR]] : !spirv.ptr<!spirv.array<[[ARRAYSIZE]] x i8>, UniformConstant> to !spirv.ptr<i8, UniformConstant>
        // CHECK-NEXT:  {{%.*}} = spirv.CL.printf [[FMTSTR_PTR1]] {{%.*}}, {{%.*}}, {{%.*}} : !spirv.ptr<i8, UniformConstant>, i32, f32, i32 -> i32
        gpu.printf "\nHello, world : %d %f \n Thread id: %d\n", %arg0, %arg1, %2: i32, f32, index

        // CHECK: spirv.Return
        gpu.return
    }
  }
}
