// RUN: mlir-opt -convert-shape-to-scf -split-input-file %s | FileCheck %s

// CHECK-LABEL: shape_reduce
// CHECK-SAME:   [[SHAPE:%.*]]: !shape.shape) -> !shape.size {
func @shape_reduce(%shape : !shape.shape) -> !shape.size {
  %init = shape.const_size 1
  %num_elements = shape.reduce(%shape, %init) -> !shape.size {
    ^bb0(%index: index, %dim: !shape.size, %acc: !shape.size):
      %new_acc = shape.mul %acc, %dim
      shape.yield %new_acc : !shape.size
  }
  return %num_elements : !shape.size
}
// CHECK-NEXT: [[SHAPE_C1:%.*]] = shape.const_size 1
// CHECK-NEXT: [[C0:%.*]] = constant 0 : index
// CHECK-NEXT: [[C1:%.*]] = constant 1 : index

// CHECK-NEXT: [[EXTENTS:%.*]] = "shape.to_extent_tensor"([[SHAPE]])
// CHECK-NEXT: [[SIZE:%.*]] = dim [[EXTENTS]], [[C0]] : tensor<?xindex>

// CHECK-NEXT: [[RESULT:%.*]] = scf.for [[I:%.*]] = [[C0]] to [[SIZE]]
// CHECK-SAME:       step [[C1]] iter_args([[ACC:%.*]] = [[SHAPE_C1]])
// CHECK-NEXT:   [[EXTENT_INDEX:%.*]] = extract_element [[EXTENTS]]{{\[}}[[I]]]
// CHECK-NEXT:   [[EXTENT:%.*]] = shape.index_to_size [[EXTENT_INDEX]]
// CHECK-NEXT:   [[NEW_ACC:%.*]] = shape.mul [[ACC]], [[EXTENT]]
// CHECK-NEXT:   scf.yield [[NEW_ACC]] : !shape.size
// CHECK-NEXT: }
// CHECK-NEXT: return [[RESULT]] : !shape.size
