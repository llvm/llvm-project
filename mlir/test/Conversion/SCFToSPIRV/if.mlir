// RUN: mlir-opt -convert-scf-to-spirv %s -o - | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @kernel_simple_selection
func.func @kernel_simple_selection(%arg2 : memref<10xf32, #spirv.storage_class<StorageBuffer>>, %arg3 : i1) {
  %value = arith.constant 0.0 : f32
  %i = arith.constant 0 : index

  // CHECK:       spirv.mlir.selection {
  // CHECK-NEXT:    spirv.BranchConditional {{%.*}}, [[TRUE:\^.*]], [[MERGE:\^.*]]
  // CHECK-NEXT:  [[TRUE]]:
  // CHECK:         spirv.Branch [[MERGE]]
  // CHECK-NEXT:  [[MERGE]]:
  // CHECK-NEXT:    spirv.mlir.merge
  // CHECK-NEXT:  }
  // CHECK-NEXT:  spirv.Return

  scf.if %arg3 {
    memref.store %value, %arg2[%i] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
  }
  return
}

// CHECK-LABEL: @kernel_nested_selection
func.func @kernel_nested_selection(%arg3 : memref<10xf32, #spirv.storage_class<StorageBuffer>>, %arg4 : memref<10xf32, #spirv.storage_class<StorageBuffer>>, %arg5 : i1, %arg6 : i1) {
  %i = arith.constant 0 : index
  %j = arith.constant 9 : index

  // CHECK:       spirv.mlir.selection {
  // CHECK-NEXT:    spirv.BranchConditional {{%.*}}, [[TRUE_TOP:\^.*]], [[FALSE_TOP:\^.*]]
  // CHECK-NEXT:  [[TRUE_TOP]]:
  // CHECK-NEXT:    spirv.mlir.selection {
  // CHECK-NEXT:      spirv.BranchConditional {{%.*}}, [[TRUE_NESTED_TRUE_PATH:\^.*]], [[FALSE_NESTED_TRUE_PATH:\^.*]]
  // CHECK-NEXT:    [[TRUE_NESTED_TRUE_PATH]]:
  // CHECK:           spirv.Branch [[MERGE_NESTED_TRUE_PATH:\^.*]]
  // CHECK-NEXT:    [[FALSE_NESTED_TRUE_PATH]]:
  // CHECK:           spirv.Branch [[MERGE_NESTED_TRUE_PATH]]
  // CHECK-NEXT:    [[MERGE_NESTED_TRUE_PATH]]:
  // CHECK-NEXT:      spirv.mlir.merge
  // CHECK-NEXT:    }
  // CHECK-NEXT:    spirv.Branch [[MERGE_TOP:\^.*]]
  // CHECK-NEXT:  [[FALSE_TOP]]:
  // CHECK-NEXT:    spirv.mlir.selection {
  // CHECK-NEXT:      spirv.BranchConditional {{%.*}}, [[TRUE_NESTED_FALSE_PATH:\^.*]], [[FALSE_NESTED_FALSE_PATH:\^.*]]
  // CHECK-NEXT:    [[TRUE_NESTED_FALSE_PATH]]:
  // CHECK:           spirv.Branch [[MERGE_NESTED_FALSE_PATH:\^.*]]
  // CHECK-NEXT:    [[FALSE_NESTED_FALSE_PATH]]:
  // CHECK:           spirv.Branch [[MERGE_NESTED_FALSE_PATH]]
  // CHECK:         [[MERGE_NESTED_FALSE_PATH]]:
  // CHECK-NEXT:      spirv.mlir.merge
  // CHECK-NEXT:    }
  // CHECK-NEXT:    spirv.Branch [[MERGE_TOP]]
  // CHECK-NEXT:  [[MERGE_TOP]]:
  // CHECK-NEXT:    spirv.mlir.merge
  // CHECK-NEXT:  }
  // CHECK-NEXT:  spirv.Return

  scf.if %arg5 {
    scf.if %arg6 {
      %value = memref.load %arg3[%i] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
      memref.store %value, %arg4[%i] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
    } else {
      %value = memref.load %arg4[%i] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
      memref.store %value, %arg3[%i] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
    }
  } else {
    scf.if %arg6 {
      %value = memref.load %arg3[%j] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
      memref.store %value, %arg4[%j] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
    } else {
      %value = memref.load %arg4[%j] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
      memref.store %value, %arg3[%j] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
    }
  }
  return
}

// CHECK-LABEL: @simple_if_yield
func.func @simple_if_yield(%arg2 : memref<10xf32, #spirv.storage_class<StorageBuffer>>, %arg3 : i1) {
  // CHECK: %[[VAR1:.*]] = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: %[[VAR2:.*]] = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK:       spirv.mlir.selection {
  // CHECK-NEXT:    spirv.BranchConditional {{%.*}}, [[TRUE:\^.*]], [[FALSE:\^.*]]
  // CHECK-NEXT:  [[TRUE]]:
  // CHECK:         %[[RET1TRUE:.*]] = spirv.Constant 0.000000e+00 : f32
  // CHECK:         %[[RET2TRUE:.*]] = spirv.Constant 1.000000e+00 : f32
  // CHECK-DAG:     spirv.Store "Function" %[[VAR1]], %[[RET1TRUE]] : f32
  // CHECK-DAG:     spirv.Store "Function" %[[VAR2]], %[[RET2TRUE]] : f32
  // CHECK:         spirv.Branch ^[[MERGE:.*]]
  // CHECK-NEXT:  [[FALSE]]:
  // CHECK:         %[[RET2FALSE:.*]] = spirv.Constant 2.000000e+00 : f32
  // CHECK:         %[[RET1FALSE:.*]] = spirv.Constant 3.000000e+00 : f32
  // CHECK-DAG:     spirv.Store "Function" %[[VAR1]], %[[RET1FALSE]] : f32
  // CHECK-DAG:     spirv.Store "Function" %[[VAR2]], %[[RET2FALSE]] : f32
  // CHECK:         spirv.Branch ^[[MERGE]]
  // CHECK-NEXT:  ^[[MERGE]]:
  // CHECK:         spirv.mlir.merge
  // CHECK-NEXT:  }
  // CHECK-DAG:   %[[OUT1:.*]] = spirv.Load "Function" %[[VAR1]] : f32
  // CHECK-DAG:   %[[OUT2:.*]] = spirv.Load "Function" %[[VAR2]] : f32
  // CHECK:       spirv.Store "StorageBuffer" {{%.*}}, %[[OUT1]] : f32
  // CHECK:       spirv.Store "StorageBuffer" {{%.*}}, %[[OUT2]] : f32
  // CHECK:       spirv.Return
  %0:2 = scf.if %arg3 -> (f32, f32) {
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    scf.yield %c0, %c1 : f32, f32
  } else {
    %c0 = arith.constant 2.0 : f32
    %c1 = arith.constant 3.0 : f32
    scf.yield %c1, %c0 : f32, f32
  }
  %i = arith.constant 0 : index
  %j = arith.constant 1 : index
  memref.store %0#0, %arg2[%i] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
  memref.store %0#1, %arg2[%j] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
  return
}

// TODO: The transformation should only be legal if VariablePointer capability
// is supported. This test is still useful to make sure we can handle scf op
// result with type change.
func.func @simple_if_yield_type_change(%arg2 : memref<10xf32, #spirv.storage_class<StorageBuffer>>, %arg3 : memref<10xf32, #spirv.storage_class<StorageBuffer>>, %arg4 : i1) {
  // CHECK-LABEL: @simple_if_yield_type_change
  // CHECK:       %[[VAR:.*]] = spirv.Variable : !spirv.ptr<!spirv.ptr<!spirv.struct<(!spirv.array<10 x f32, stride=4> [0])>, StorageBuffer>, Function>
  // CHECK:       spirv.mlir.selection {
  // CHECK-NEXT:    spirv.BranchConditional {{%.*}}, [[TRUE:\^.*]], [[FALSE:\^.*]]
  // CHECK-NEXT:  [[TRUE]]:
  // CHECK:         spirv.Store "Function" %[[VAR]], {{%.*}} : !spirv.ptr<!spirv.struct<(!spirv.array<10 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK:         spirv.Branch ^[[MERGE:.*]]
  // CHECK-NEXT:  [[FALSE]]:
  // CHECK:         spirv.Store "Function" %[[VAR]], {{%.*}} : !spirv.ptr<!spirv.struct<(!spirv.array<10 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK:         spirv.Branch ^[[MERGE]]
  // CHECK-NEXT:  ^[[MERGE]]:
  // CHECK:         spirv.mlir.merge
  // CHECK-NEXT:  }
  // CHECK:       %[[OUT:.*]] = spirv.Load "Function" %[[VAR]] : !spirv.ptr<!spirv.struct<(!spirv.array<10 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK:       %[[ADD:.*]] = spirv.AccessChain %[[OUT]][{{%.*}}, {{%.*}}] : !spirv.ptr<!spirv.struct<(!spirv.array<10 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK:       spirv.Store "StorageBuffer" %[[ADD]], {{%.*}} : f32
  // CHECK:       spirv.Return
  %i = arith.constant 0 : index
  %value = arith.constant 0.0 : f32
  %0 = scf.if %arg4 -> (memref<10xf32, #spirv.storage_class<StorageBuffer>>) {
    scf.yield %arg2 : memref<10xf32, #spirv.storage_class<StorageBuffer>>
  } else {
    scf.yield %arg3 : memref<10xf32, #spirv.storage_class<StorageBuffer>>
  }
  memref.store %value, %0[%i] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
  return
}

// Memrefs without a spirv storage class are not supported. The conversion
// should preserve the `scf.if` and not crash.
func.func @unsupported_yield_type(%arg0 : memref<8xi32>, %arg1 : memref<8xi32>, %c : i1) {
// CHECK-LABEL: @unsupported_yield_type
// CHECK-NEXT:    scf.if
// CHECK:         spirv.Return
  %r = scf.if %c -> (memref<8xi32>) {
    scf.yield %arg0 : memref<8xi32>
  } else {
    scf.yield %arg1 : memref<8xi32>
  }
  return
}

} // end module
