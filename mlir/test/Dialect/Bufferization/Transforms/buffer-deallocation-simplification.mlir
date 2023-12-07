// RUN: mlir-opt %s --buffer-deallocation-simplification --split-input-file | FileCheck %s

func.func @dealloc_deallocated_in_retained(%arg0: memref<2xi32>, %arg1: i1, %arg2: memref<2xi32>, %arg3: i1) -> (i1, i1, i1, i1, i1, i1, i1) {
  %0 = bufferization.dealloc (%arg0 : memref<2xi32>) if (%arg1) retain (%arg0 : memref<2xi32>)
  %1 = bufferization.dealloc (%arg0, %arg2 : memref<2xi32>, memref<2xi32>) if (%arg1, %arg1) retain (%arg0 : memref<2xi32>)
  %2:2 = bufferization.dealloc (%arg0 : memref<2xi32>) if (%arg1) retain (%arg0, %arg2 : memref<2xi32>, memref<2xi32>)
  // multiple must-alias
  %3 = memref.subview %arg0[0][1][1] : memref<2xi32> to memref<i32>
  %4 = memref.subview %arg0[1][1][1] : memref<2xi32> to memref<1xi32, strided<[1], offset: 1>>
  %alloc = memref.alloc() : memref<2xi32>
  %5:3 = bufferization.dealloc (%arg0, %4 : memref<2xi32>, memref<1xi32, strided<[1], offset: 1>>) if (%arg1, %arg3) retain (%arg0, %alloc, %3 : memref<2xi32>, memref<2xi32>, memref<i32>)
  return %0, %1, %2#0, %2#1, %5#0, %5#1, %5#2 : i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: func @dealloc_deallocated_in_retained
//  CHECK-SAME: ([[ARG0:%.+]]: memref<2xi32>, [[ARG1:%.+]]: i1, [[ARG2:%.+]]: memref<2xi32>, [[ARG3:%.+]]: i1)
//  CHECK-NEXT: arith.constant false
//  CHECK-NEXT: [[V1:%.+]] = bufferization.dealloc ([[ARG2]] : memref<2xi32>) if ([[ARG1]]) retain ([[ARG0]] : memref<2xi32>)
//  CHECK-NEXT: [[O1:%.+]] = arith.ori [[V1]], [[ARG1]]
//  CHECK-NEXT: [[V2:%.+]]:2 = bufferization.dealloc ([[ARG0]] : memref<2xi32>) if ([[ARG1]]) retain ([[ARG0]], [[ARG2]] : memref<2xi32>, memref<2xi32>)
// COM: the RemoveRetainedMemrefsGuaranteedToNotAlias pattern removes all the
// COM: retained memrefs since the list of memrefs to be deallocated becomes empty
// COM: due to the pattern under test (and thus there is no memref the retain values
// COM: could alias to)
// CHECK-NOT: if
//  CHECK-NEXT: [[V3:%.+]] = arith.ori [[ARG3]], [[ARG1]]
//  CHECK-NEXT: [[V4:%.+]] = arith.ori [[ARG3]], [[ARG1]]
//  CHECK-NEXT: return [[ARG1]], [[O1]], [[V2]]#0, [[V2]]#1, [[V3]], %false{{[0-9_]*}}, [[V4]] :

// -----

func.func @dealloc_deallocated_in_retained_extract_base_memref(%arg0: memref<2xi32>, %arg1: i1, %arg2: memref<2xi32>, %arg3: i1) -> (i1, i1, i1, i1, i1, i1, i1) {
  %base_buffer, %offset, %size, %stride = memref.extract_strided_metadata %arg0 : memref<2xi32> -> memref<i32>, index, index, index
  %base_buffer0, %offset0, %size0, %stride0 = memref.extract_strided_metadata %arg2 : memref<2xi32> -> memref<i32>, index, index, index
  %0 = bufferization.dealloc (%base_buffer : memref<i32>) if (%arg1) retain (%arg0 : memref<2xi32>)
  %1 = bufferization.dealloc (%base_buffer, %base_buffer0 : memref<i32>, memref<i32>) if (%arg1, %arg1) retain (%arg0 : memref<2xi32>)
  %2:2 = bufferization.dealloc (%base_buffer : memref<i32>) if (%arg1) retain (%arg0, %arg2 : memref<2xi32>, memref<2xi32>)
  // multiple must-alias
  %3 = memref.subview %arg0[0][1][1] : memref<2xi32> to memref<i32>
  %4 = memref.subview %arg0[1][1][1] : memref<2xi32> to memref<1xi32, strided<[1], offset: 1>>
  %alloc = memref.alloc() : memref<2xi32>
  %5:3 = bufferization.dealloc (%base_buffer, %4 : memref<i32>, memref<1xi32, strided<[1], offset: 1>>) if (%arg1, %arg3) retain (%arg0, %alloc, %3 : memref<2xi32>, memref<2xi32>, memref<i32>)
  return %0, %1, %2#0, %2#1, %5#0, %5#1, %5#2 : i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: func @dealloc_deallocated_in_retained_extract_base_memref
//  CHECK-SAME: ([[ARG0:%.+]]: memref<2xi32>, [[ARG1:%.+]]: i1, [[ARG2:%.+]]: memref<2xi32>, [[ARG3:%.+]]: i1)
//  CHECK-NEXT: arith.constant false
//  CHECK-NEXT: [[BASE0:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[ARG0]] :
//  CHECK-NEXT: [[BASE1:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[ARG2]] :
//  CHECK-NEXT: [[V1:%.+]] = bufferization.dealloc ([[BASE1]] : memref<i32>) if ([[ARG1]]) retain ([[ARG0]] : memref<2xi32>)
//  CHECK-NEXT: [[O1:%.+]] = arith.ori [[V1]], [[ARG1]]
//  CHECK-NEXT: [[V2:%.+]]:2 = bufferization.dealloc ([[BASE0]] : memref<i32>) if ([[ARG1]]) retain ([[ARG0]], [[ARG2]] : memref<2xi32>, memref<2xi32>)
// COM: the RemoveRetainedMemrefsGuaranteedToNotAlias pattern removes all the
// COM: retained memrefs since the list of memrefs to be deallocated becomes empty
// COM: due to the pattern under test (and thus there is no memref the retain values
// COM: could alias to)
// CHECK-NOT: if
//  CHECK-NEXT: [[V3:%.+]] = arith.ori [[ARG3]], [[ARG1]]
//  CHECK-NEXT: [[V4:%.+]] = arith.ori [[ARG3]], [[ARG1]]
//  CHECK-NEXT: return [[ARG1]], [[O1]], [[V2]]#0, [[V2]]#1, [[V3]], %false{{[0-9_]*}}, [[V4]] :

// -----

func.func @remove_retained_memrefs_guarateed_to_not_alias(%arg0: i1, %arg1: memref<2xi32>) -> (i1, i1, memref<2xi32>) {
  %alloc = memref.alloc() : memref<2xi32>
  %alloc0 = memref.alloc() : memref<2xi32>
  %0:2 = bufferization.dealloc (%alloc : memref<2xi32>) if (%arg0) retain (%alloc0, %arg1 : memref<2xi32>, memref<2xi32>)
  return %0#0, %0#1, %alloc : i1, i1, memref<2xi32>
}

// CHECK-LABEL: func @remove_retained_memrefs_guarateed_to_not_alias
//  CHECK-SAME: ([[ARG0:%.+]]: i1, [[ARG1:%.+]]: memref<2xi32>)
//  CHECK-NEXT: [[FALSE:%.+]] = arith.constant false
//  CHECK-NEXT: [[ALLOC:%.+]] = memref.alloc(
//  CHECK-NEXT: bufferization.dealloc ([[ALLOC]] : memref<2xi32>) if ([[ARG0]])
//  CHECK-NOT: retain
//  CHECK-NEXT: return [[FALSE]], [[FALSE]], [[ALLOC]] :

// -----

func.func @dealloc_split_when_no_other_aliasing(%arg0: i1, %arg1: memref<2xi32>, %arg2: memref<2xi32>, %arg3: i1) -> (i1, i1) {
  %alloc = memref.alloc() : memref<2xi32>
  %alloc0 = memref.alloc() : memref<2xi32>
  %0 = arith.select %arg0, %alloc, %alloc0 : memref<2xi32>
  %1:2 = bufferization.dealloc (%alloc, %arg2 : memref<2xi32>, memref<2xi32>) if (%arg0, %arg3) retain (%arg1, %0 : memref<2xi32>, memref<2xi32>)
  return %1#0, %1#1 : i1, i1
}

// CHECK-LABEL: func @dealloc_split_when_no_other_aliasing
//  CHECK-SAME: ([[ARG0:%.+]]: i1, [[ARG1:%.+]]: memref<2xi32>, [[ARG2:%.+]]: memref<2xi32>, [[ARG3:%.+]]: i1)
//  CHECK-NEXT:   [[ALLOC0:%.+]] = memref.alloc(
//  CHECK-NEXT:   [[ALLOC1:%.+]] = memref.alloc(
//  CHECK-NEXT:   [[V0:%.+]] = arith.select{{.*}}[[ALLOC0]], [[ALLOC1]] :
// COM: there is only one value in the retained list because the
// COM: RemoveRetainedMemrefsGuaranteedToNotAlias pattern also applies here and
// COM: removes %arg1 from the list. In the second dealloc, this does not apply
// COM: because function arguments are assumed potentially alias (even if the
// COM: types don't exactly match).
//  CHECK-NEXT:   [[V1:%.+]] = bufferization.dealloc ([[ALLOC0]] : memref<2xi32>) if ([[ARG0]]) retain ([[V0]] : memref<2xi32>)
//  CHECK-NEXT:   [[V2:%.+]]:2 = bufferization.dealloc ([[ARG2]] : memref<2xi32>) if ([[ARG3]]) retain ([[ARG1]], [[V0]] : memref<2xi32>, memref<2xi32>)
//  CHECK-NEXT:   [[V3:%.+]] = arith.ori [[V1]], [[V2]]#1
//  CHECK-NEXT:   return [[V2]]#0, [[V3]] :

// -----

func.func @dealloc_remove_dealloc_memref_contained_in_retained_with_const_true_condition(
  %arg0: memref<2xi32>, %arg1: memref<2xi32>, %arg2: memref<2xi32>) -> (memref<2xi32>, memref<2xi32>, i1, i1) {
  %true = arith.constant true
  %0:2 = bufferization.dealloc (%arg0, %arg1, %arg2 : memref<2xi32>, memref<2xi32>, memref<2xi32>) if (%true, %true, %true) retain (%arg0, %arg1 : memref<2xi32>, memref<2xi32>)
  return %arg0, %arg1, %0#0, %0#1 : memref<2xi32>, memref<2xi32>, i1, i1
}

// CHECK-LABEL: func @dealloc_remove_dealloc_memref_contained_in_retained_with_const_true_condition
//  CHECK-SAME: ([[ARG0:%.+]]: memref<2xi32>, [[ARG1:%.+]]: memref<2xi32>, [[ARG2:%.+]]: memref<2xi32>)
//       CHECK:   bufferization.dealloc ([[ARG2]] :{{.*}}) if (%true{{[0-9_]*}})
//  CHECK-NEXT:   return [[ARG0]], [[ARG1]], %true{{[0-9_]*}}, %true{{[0-9_]*}} :

// -----

func.func @dealloc_remove_dealloc_memref_contained_in_retained_with_const_true_condition(
  %arg0: memref<2xi32>, %arg1: memref<2xi32>, %arg2: memref<2xi32>) -> (memref<2xi32>, memref<2xi32>, i1, i1) {
  %true = arith.constant true
  %base_buffer, %offset, %size, %stride = memref.extract_strided_metadata %arg0 : memref<2xi32> -> memref<i32>, index, index, index
  %base_buffer_1, %offset_1, %size_1, %stride_1 = memref.extract_strided_metadata %arg1 : memref<2xi32> -> memref<i32>, index, index, index
  %base_buffer_2, %offset_2, %size_2, %stride_2 = memref.extract_strided_metadata %arg2 : memref<2xi32> -> memref<i32>, index, index, index
  %0:2 = bufferization.dealloc (%base_buffer, %base_buffer_1, %base_buffer_2 : memref<i32>, memref<i32>, memref<i32>) if (%true, %true, %true) retain (%arg0, %arg1 : memref<2xi32>, memref<2xi32>)
  return %arg0, %arg1, %0#0, %0#1 : memref<2xi32>, memref<2xi32>, i1, i1
}

// CHECK-LABEL: func @dealloc_remove_dealloc_memref_contained_in_retained_with_const_true_condition
//  CHECK-SAME: ([[ARG0:%.+]]: memref<2xi32>, [[ARG1:%.+]]: memref<2xi32>, [[ARG2:%.+]]: memref<2xi32>)
//       CHECK:   [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[ARG2]]
//       CHECK:   bufferization.dealloc ([[BASE]] :{{.*}}) if (%true{{[0-9_]*}})
//  CHECK-NEXT:   return [[ARG0]], [[ARG1]], %true{{[0-9_]*}}, %true{{[0-9_]*}} :
