// RUN: mlir-opt -gpu-serialize-to-spirv %s | FileCheck %s
module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spirv.resource_limits<>>} {
  // CHECK:        gpu.module @addt_kernel attributes {gpu.binary =
  spirv.module @__spv__addt_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Addresses, Kernel], []> {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @addt_kernel(%arg0: !spirv.ptr<f32, CrossWorkgroup>, %arg1: !spirv.ptr<f32, CrossWorkgroup>, %arg2: !spirv.ptr<f32, CrossWorkgroup>) "None" attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>, workgroup_attributions = 0 : i64} {
      %cst5_i64 = spirv.Constant 5 : i64
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
      spirv.Branch ^bb1
    ^bb1:  // pred: ^bb0
      %4 = spirv.IMul %1, %cst5_i64 : i64
      %5 = spirv.IAdd %4, %3 : i64
      %6 = spirv.InBoundsPtrAccessChain %arg0[%5] : !spirv.ptr<f32, CrossWorkgroup>, i64
      %7 = spirv.Load "CrossWorkgroup" %6 ["Aligned", 4] : f32
      %8 = spirv.IMul %1, %cst5_i64 : i64
      %9 = spirv.IAdd %8, %3 : i64
      %10 = spirv.InBoundsPtrAccessChain %arg1[%9] : !spirv.ptr<f32, CrossWorkgroup>, i64
      %11 = spirv.Load "CrossWorkgroup" %10 ["Aligned", 4] : f32
      %12 = spirv.FAdd %7, %11 : f32
      %13 = spirv.IMul %1, %cst5_i64 : i64
      %14 = spirv.IAdd %13, %3 : i64
      %15 = spirv.InBoundsPtrAccessChain %arg2[%14] : !spirv.ptr<f32, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %15, %12 ["Aligned", 4] : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @addt_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @addt_kernel {
    gpu.func @addt_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c5 = arith.constant 5 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %2 = arith.muli %0, %c5 : index
      %3 = arith.addi %2, %1 : index
      %4 = memref.load %arg0[%3] : memref<?xf32>
      %5 = arith.muli %0, %c5 : index
      %6 = arith.addi %5, %1 : index
      %7 = memref.load %arg1[%6] : memref<?xf32>
      %8 = arith.addf %4, %7 : f32
      %9 = arith.muli %0, %c5 : index
      %10 = arith.addi %9, %1 : index
      memref.store %8, %arg2[%10] : memref<?xf32>
      gpu.return
    }
  }
}
