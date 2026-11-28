// RUN: mlir-opt -split-input-file -spirv-lower-abi-attrs %s | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Kernel, Addresses], []>, #spirv.resource_limits<>>
} {
  spirv.module Physical64 OpenCL {
    // CHECK-LABEL: spirv.module
    //       CHECK:   spirv.func [[FN:@.*]]({{%.*}}: f32, {{%.*}}: !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32>)>, CrossWorkgroup>
    // We cannot generate SubgroupSize execution mode without necessary capability -- leave it alone.
    //  CHECK-SAME:      #spirv.entry_point_abi<subgroup_size = 64>
    //       CHECK:   spirv.EntryPoint "Kernel" [[FN]]
    //       CHECK:   spirv.ExecutionMode [[FN]] "LocalSize", 32, 1, 1
    spirv.func @kernel(
      %arg0: f32,
      %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32>)>, CrossWorkgroup>) "None"
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1], subgroup_size = 64>} {
      spirv.Return
    }
  }
}

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Kernel, SubgroupDispatch], []>, #spirv.resource_limits<>>
} {
  spirv.module Physical64 OpenCL {
    // CHECK-LABEL: spirv.module
    //       CHECK:   spirv.func [[FN:@.*]]({{%.*}}: f32, {{%.*}}: !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32>)>, CrossWorkgroup>
    //       CHECK:   spirv.EntryPoint "Kernel" [[FN]]
    //       CHECK:   spirv.ExecutionMode [[FN]] "LocalSize", 32, 1, 1
    //       CHECK:   spirv.ExecutionMode [[FN]] "SubgroupSize", 64
    spirv.func @kernel(
      %arg0: f32,
      %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32>)>, CrossWorkgroup>) "None"
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1], subgroup_size = 64>} {
      spirv.Return
    }
  }
}

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses, UntypedPointersKHR], [SPV_KHR_untyped_pointers]>, #spirv.resource_limits<>>
} {
  spirv.module Physical64 OpenCL {
    // CHECK-LABEL: spirv.module
    spirv.GlobalVariable @var data_type(f32) : !spirv.untyped_ptr<Workgroup>
    //       CHECK:   spirv.func [[FN:@.*]]()
    spirv.func @kernel() "None"
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1]>} {
      %0 = spirv.mlir.addressof @var : !spirv.untyped_ptr<Workgroup>
      spirv.Return
    }
    //       CHECK:   spirv.EntryPoint "Kernel" [[FN]], @var
  }
}
