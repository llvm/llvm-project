// RUN: mlir-opt -convert-scf-to-spirv %s -o - | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @switch_no_result
func.func @switch_no_result(%arg0 : memref<10xf32, #spirv.storage_class<StorageBuffer>>, %arg1 : index) {
  %value = arith.constant 0.0 : f32
  %i = arith.constant 0 : index

  // CHECK:       spirv.mlir.selection {
  // CHECK-NEXT:    spirv.Switch {{%.*}} : i32, [
  // CHECK-NEXT:      default: [[DEFAULT:\^.*]],
  // CHECK-NEXT:      2: [[CASE:\^.*]]
  // CHECK-NEXT:    ]
  // CHECK-NEXT:  [[CASE]]:
  // CHECK:         spirv.Branch [[MERGE:\^.*]]
  // CHECK-NEXT:  [[DEFAULT]]:
  // CHECK-NEXT:    spirv.Branch [[MERGE]]
  // CHECK-NEXT:  [[MERGE]]:
  // CHECK-NEXT:    spirv.mlir.merge
  // CHECK-NEXT:  }
  // CHECK-NEXT:  spirv.Return

  scf.index_switch %arg1
  case 2 {
    memref.store %value, %arg0[%i] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
    scf.yield
  }
  default {
    scf.yield
  }
  return
}

// CHECK-LABEL: @switch_yield
func.func @switch_yield(%arg1 : index) -> i32 {
  // CHECK:       %[[VAR:.*]] = spirv.Variable : !spirv.ptr<i32, Function>
  // CHECK:       spirv.mlir.selection {
  // CHECK-NEXT:    spirv.Switch {{%.*}} : i32, [
  // CHECK-NEXT:      default: [[DEFAULT:\^.*]],
  // CHECK-NEXT:      2: [[CASE2:\^.*]],
  // CHECK-NEXT:      5: [[CASE5:\^.*]]
  // CHECK-NEXT:    ]
  // CHECK-NEXT:  [[CASE2]]:
  // CHECK:         %[[C10:.*]] = spirv.Constant 10 : i32
  // CHECK:         spirv.Store "Function" %[[VAR]], %[[C10]] : i32
  // CHECK:         spirv.Branch [[MERGE:\^.*]]
  // CHECK-NEXT:  [[CASE5]]:
  // CHECK:         %[[C20:.*]] = spirv.Constant 20 : i32
  // CHECK:         spirv.Store "Function" %[[VAR]], %[[C20]] : i32
  // CHECK:         spirv.Branch [[MERGE]]
  // CHECK-NEXT:  [[DEFAULT]]:
  // CHECK:         %[[C30:.*]] = spirv.Constant 30 : i32
  // CHECK:         spirv.Store "Function" %[[VAR]], %[[C30]] : i32
  // CHECK:         spirv.Branch [[MERGE]]
  // CHECK-NEXT:  [[MERGE]]:
  // CHECK-NEXT:    spirv.mlir.merge
  // CHECK-NEXT:  }
  // CHECK:       %[[OUT:.*]] = spirv.Load "Function" %[[VAR]] : i32
  // CHECK:       spirv.ReturnValue %[[OUT]] : i32
  %0 = scf.index_switch %arg1 -> i32
  case 2 {
    %c10 = arith.constant 10 : i32
    scf.yield %c10 : i32
  }
  case 5 {
    %c20 = arith.constant 20 : i32
    scf.yield %c20 : i32
  }
  default {
    %c30 = arith.constant 30 : i32
    scf.yield %c30 : i32
  }
  return %0 : i32
}

// CHECK-LABEL: @switch_selection_control
func.func @switch_selection_control(%arg0 : memref<10xf32, #spirv.storage_class<StorageBuffer>>, %arg1 : index) {
  %value = arith.constant 0.0 : f32
  %i = arith.constant 0 : index
  // CHECK: spirv.mlir.selection control(Flatten) {
  scf.index_switch %arg1 {spirv.selection_control = #spirv.selection_control<Flatten>}
  case 2 {
    memref.store %value, %arg0[%i] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
    scf.yield
  }
  default {
    scf.yield
  }
  return
}

// A case value outside the 32-bit selector would be truncated onto another
// value: the switch is not converted.
// CHECK-LABEL: @switch_case_out_of_selector_range
// CHECK-NOT:     spirv.Switch
// CHECK:         scf.index_switch
// CHECK-NEXT:    case 4294967296
func.func @switch_case_out_of_selector_range(%arg0 : index) {
  scf.index_switch %arg0
  case 4294967296 {
    scf.yield
  }
  default {
    scf.yield
  }
  return
}

// Here 4294967296 would have collided with case 0.
// CHECK-LABEL: @switch_case_truncates_onto_another
// CHECK-NOT:     spirv.Switch
// CHECK:         scf.index_switch
func.func @switch_case_truncates_onto_another(%arg0 : index) {
  scf.index_switch %arg0
  case 0 {
    scf.yield
  }
  case 4294967296 {
    scf.yield
  }
  default {
    scf.yield
  }
  return
}

// The bounds of the 32-bit selector still convert.
// CHECK-LABEL: @switch_case_selector_bounds
// CHECK:         spirv.Switch {{%.*}} : i32, [
// CHECK-NEXT:      default: {{\^.*}},
// CHECK-NEXT:      2147483647: {{\^.*}},
// CHECK-NEXT:      -2147483648: {{\^.*}}
func.func @switch_case_selector_bounds(%arg0 : index) {
  scf.index_switch %arg0
  case 2147483647 {
    scf.yield
  }
  case -2147483648 {
    scf.yield
  }
  default {
    scf.yield
  }
  return
}

} // end module
