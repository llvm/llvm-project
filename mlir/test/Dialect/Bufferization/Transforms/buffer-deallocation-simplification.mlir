// RUN: mlir-opt %s --buffer-deallocation-simplification --split-input-file | FileCheck %s

func.func @dealloc_deallocated_in_retained(%arg0: memref<2xi32>, %arg1: i1, %arg2: memref<2xi32>) -> (i1, i1, i1, i1) {
  %0 = bufferization.dealloc (%arg0 : memref<2xi32>) if (%arg1) retain (%arg0 : memref<2xi32>)
  %1 = bufferization.dealloc (%arg0, %arg2 : memref<2xi32>, memref<2xi32>) if (%arg1, %arg1) retain (%arg0 : memref<2xi32>)
  %2:2 = bufferization.dealloc (%arg0 : memref<2xi32>) if (%arg1) retain (%arg0, %arg2 : memref<2xi32>, memref<2xi32>)
  return %0, %1, %2#0, %2#1 : i1, i1, i1, i1
}

// CHECK-LABEL: func @dealloc_deallocated_in_retained
//  CHECK-SAME: ([[ARG0:%.+]]: memref<2xi32>, [[ARG1:%.+]]: i1, [[ARG2:%.+]]: memref<2xi32>)
//  CHECK-NEXT: [[V0:%.+]] = bufferization.dealloc retain ([[ARG0]] : memref<2xi32>)
//  CHECK-NEXT: [[O0:%.+]] = arith.ori [[V0]], [[ARG1]]
//  CHECK-NEXT: [[V1:%.+]] = bufferization.dealloc ([[ARG2]] : memref<2xi32>) if ([[ARG1]]) retain ([[ARG0]] : memref<2xi32>)
//  CHECK-NEXT: [[O1:%.+]] = arith.ori [[V1]], [[ARG1]]
//  CHECK-NEXT: [[V2:%.+]]:2 = bufferization.dealloc ([[ARG0]] : memref<2xi32>) if ([[ARG1]]) retain ([[ARG0]], [[ARG2]] : memref<2xi32>, memref<2xi32>)
//  CHECK-NEXT: return [[O0]], [[O1]], [[V2]]#0, [[V2]]#1 :

// -----

func.func @dealloc_deallocated_in_retained_extract_base_memref(%arg0: memref<2xi32>, %arg1: i1, %arg2: memref<2xi32>) -> (i1, i1, i1, i1) {
  %base_buffer, %offset, %size, %stride = memref.extract_strided_metadata %arg0 : memref<2xi32> -> memref<i32>, index, index, index
  %base_buffer0, %offset0, %size0, %stride0 = memref.extract_strided_metadata %arg2 : memref<2xi32> -> memref<i32>, index, index, index
  %0 = bufferization.dealloc (%base_buffer : memref<i32>) if (%arg1) retain (%arg0 : memref<2xi32>)
  %1 = bufferization.dealloc (%base_buffer, %base_buffer0 : memref<i32>, memref<i32>) if (%arg1, %arg1) retain (%arg0 : memref<2xi32>)
  %2:2 = bufferization.dealloc (%base_buffer : memref<i32>) if (%arg1) retain (%arg0, %arg2 : memref<2xi32>, memref<2xi32>)
  return %0, %1, %2#0, %2#1 : i1, i1, i1, i1
}

// CHECK-LABEL: func @dealloc_deallocated_in_retained_extract_base_memref
//  CHECK-SAME: ([[ARG0:%.+]]: memref<2xi32>, [[ARG1:%.+]]: i1, [[ARG2:%.+]]: memref<2xi32>)
//  CHECK-NEXT: [[BASE0:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[ARG0]] :
//  CHECK-NEXT: [[BASE1:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[ARG2]] :
//  CHECK-NEXT: [[V0:%.+]] = bufferization.dealloc retain ([[ARG0]] : memref<2xi32>)
//  CHECK-NEXT: [[O0:%.+]] = arith.ori [[V0]], [[ARG1]]
//  CHECK-NEXT: [[V1:%.+]] = bufferization.dealloc ([[BASE1]] : memref<i32>) if ([[ARG1]]) retain ([[ARG0]] : memref<2xi32>)
//  CHECK-NEXT: [[O1:%.+]] = arith.ori [[V1]], [[ARG1]]
//  CHECK-NEXT: [[V2:%.+]]:2 = bufferization.dealloc ([[BASE0]] : memref<i32>) if ([[ARG1]]) retain ([[ARG0]], [[ARG2]] : memref<2xi32>, memref<2xi32>)
//  CHECK-NEXT: return [[O0]], [[O1]], [[V2]]#0, [[V2]]#1 :
