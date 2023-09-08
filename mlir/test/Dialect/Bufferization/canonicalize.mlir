// RUN: mlir-opt %s \
// RUN:   -canonicalize="test-convergence" \
// RUN:   --split-input-file -allow-unregistered-dialect | \
// RUN: FileCheck %s

// Basic folding of to_tensor(to_memref(t)) -> t
// CHECK-LABEL: func @tensor_load_of_buffer_cast(
func.func @tensor_load_of_buffer_cast(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = bufferization.to_memref %arg0 : memref<?xf32>
  %1 = bufferization.to_tensor %0 : memref<?xf32>
  return %1 : tensor<?xf32>
}
// CHECK-SAME:   %[[TENSOR:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK: return %[[TENSOR]]

// -----

// Basic folding of to_memref(to_tensor(m)) -> m
// CHECK-LABEL: func @buffer_cast_of_tensor_load(
func.func @buffer_cast_of_tensor_load(%arg0: memref<?xf32>) -> memref<?xf32> {
  %0 = bufferization.to_tensor %arg0 : memref<?xf32>
  %1 = bufferization.to_memref %0 : memref<?xf32>
  return %1 : memref<?xf32>
}
// CHECK-SAME:   %[[MEMREF:.*]]: memref<?xf32>) -> memref<?xf32> {
// CHECK: return %[[MEMREF]]

// -----

// If the memrefs are not the same type, don't fold them.
// If the memrefs are not cast-compatible (e.g. different address space), don't
// canonicalize them either.
// CHECK-LABEL: func @no_fold_buffer_cast_of_tensor_load(
//  CHECK-SAME:   %[[MEMREF_ADDRSPACE2:.*]]: memref<?xf32, 2>)
//  CHECK-SAME:     -> memref<?xf32, 7> {
//       CHECK: %[[TENSOR:.*]] = bufferization.to_tensor
//  CHECK-SAME:   %[[MEMREF_ADDRSPACE2]] : memref<?xf32, 2>
//       CHECK: %[[MEMREF_ADDRSPACE7:.*]] = bufferization.to_memref
//  CHECK-SAME:   %[[TENSOR]] : memref<?xf32, 7>
//       CHECK: return %[[MEMREF_ADDRSPACE7]]
func.func @no_fold_buffer_cast_of_tensor_load(%arg0: memref<?xf32, 2>)
    -> memref<?xf32, 7> {
  %0 = bufferization.to_tensor %arg0 : memref<?xf32, 2>
  %1 = bufferization.to_memref %0 : memref<?xf32, 7>
  return %1 : memref<?xf32, 7>
}

// -----

// If the memrefs are definitely cast-compatible, canonicalize to
//            cast.
// CHECK-LABEL: func @canonicalize_buffer_cast_of_tensor_load(
//  CHECK-SAME:   %[[M:.*]]: memref<?xf32, strided<[1], offset: 3>>)
//  CHECK-SAME:     -> memref<?xf32, strided<[1], offset: ?>> {
//   CHECK-NOT: bufferization.to_tensor
//   CHECK-NOT: bufferization.to_memref
//       CHECK: %[[R:.*]] = memref.cast %[[M]]
//  CHECK-SAME:   memref<?xf32, strided<[1], offset: 3>> to memref<?xf32, strided<[1], offset: ?>>
//       CHECK: return %[[R]]
func.func @canonicalize_buffer_cast_of_tensor_load(
  %arg0: memref<?xf32, strided<[1], offset: 3>>)
  -> memref<?xf32, strided<[1], offset: ?>>
{
  %0 = bufferization.to_tensor %arg0 : memref<?xf32, strided<[1], offset: 3>>
  %1 = bufferization.to_memref %0 : memref<?xf32, strided<[1], offset: ?>>
  return %1 : memref<?xf32, strided<[1], offset: ?>>
}

// -----

// If the memrefs are potentially cast-compatible, canonicalize to
//            copy.
// CHECK-LABEL: func @canonicalize_buffer_cast_of_tensor_load_to_copy(
func.func @canonicalize_buffer_cast_of_tensor_load_to_copy(
  %arg0: memref<?xf32, strided<[1], offset: ?>>)
  -> memref<?xf32, strided<[1], offset: 3>> {
  %0 = bufferization.to_tensor %arg0 : memref<?xf32, strided<[1], offset: ?>>
  %1 = bufferization.to_memref %0 : memref<?xf32, strided<[1], offset: 3>>
  return %1 : memref<?xf32, strided<[1], offset: 3>>
}
// CHECK-SAME:   %[[M:.*]]: memref<?xf32, strided<[1], offset: ?>>)
// CHECK-SAME:     -> memref<?xf32, strided<[1], offset: 3>> {
//  CHECK-NOT: bufferization.to_tensor
//  CHECK-NOT: bufferization.to_memref
//      CHECK: %[[C0:.*]] = arith.constant 0 : index
//      CHECK: %[[DIM:.*]] = memref.dim %[[M]], %[[C0]] : memref<?xf32, strided<[1], offset: ?>>
//      CHECK: %[[ALLOC:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32, strided<[1], offset: 3>>
//      CHECK: memref.copy %[[M]], %[[ALLOC]]
// CHECK-SAME:   memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: 3>>
//      CHECK: return %[[ALLOC]]

// -----


// Basic folding of tensor.dim(to_tensor(m)) -> memref.dim(m).
// CHECK-LABEL: func @dim_of_tensor_load(
//  CHECK-SAME:     %[[MEMREF:[0-9a-z]*]]: memref<?xf32>
//       CHECK:   %[[C0:.*]] = arith.constant 0
//       CHECK:   %[[D:.*]] = memref.dim %[[MEMREF]], %[[C0]]
//       CHECK:   return %[[D]] : index
func.func @dim_of_tensor_load(%arg0: memref<?xf32>) -> index {
  %c0 = arith.constant 0 : index
  %0 = bufferization.to_tensor %arg0 : memref<?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?xf32>
  return %1 : index
}

// -----

// CHECK-LABEL: @clone_before_dealloc
func.func @clone_before_dealloc(%arg0: memref<?xf32>) -> memref<?xf32> {
  %0 = bufferization.clone %arg0 : memref<?xf32> to memref<?xf32>
  memref.dealloc %arg0 : memref<?xf32>
  return %0 : memref<?xf32>
}
// CHECK-SAME: %[[ARG:.*]]: memref<?xf32>
// CHECK-NEXT: return %[[ARG]]

// -----

// CHECK-LABEL: @clone_before_dealloc
func.func @clone_before_dealloc(%arg0: memref<?xf32>) -> memref<?xf32> {
  %0 = bufferization.clone %arg0 : memref<?xf32> to memref<?xf32>
  "use"(%0) : (memref<?xf32>) -> ()
  memref.dealloc %0 : memref<?xf32>
  return %arg0 : memref<?xf32>
}
// CHECK-SAME: %[[ARG:.*]]: memref<?xf32>
// CHECK-NEXT: "use"(%arg0)
// CHECK-NEXT: return %[[ARG]]

// -----

// CHECK-LABEL: @clone_after_cast
func.func @clone_after_cast(%arg0: memref<?xf32>) -> memref<32xf32> {
  %0 = memref.cast %arg0 : memref<?xf32> to memref<32xf32>
  %1 = bufferization.clone %0 : memref<32xf32> to memref<32xf32>
  return %1 : memref<32xf32>
}
// CHECK-SAME: %[[ARG:.*]]: memref<?xf32>
// CHECK-NEXT: bufferization.clone %[[ARG]] : memref<?xf32> to memref<32xf32>
// CHECK-NOT: memref.cast

// -----

// CHECK-LABEL: @clone_and_cast
func.func @clone_and_cast(%arg0: memref<?xf32>) -> memref<32xf32> {
  %0 = bufferization.clone %arg0 : memref<?xf32> to memref<32xf32>
  memref.dealloc %arg0 : memref<?xf32>
  return %0 : memref<32xf32>
}
// CHECK-SAME: %[[ARG:.*]]: memref<?xf32>
// CHECK-NEXT: %[[RES:.*]] = memref.cast %[[ARG]]
// CHECK-SAME:   memref<?xf32> to memref<32xf32>
// CHECK-NEXT: return %[[RES]]

// -----

// CHECK-LABEL: @alias_is_freed
func.func @alias_is_freed(%arg0 : memref<?xf32>) {
  %0 = memref.cast %arg0 : memref<?xf32> to memref<32xf32>
  %1 = bufferization.clone %0 : memref<32xf32> to memref<32xf32>
  memref.dealloc %arg0 : memref<?xf32>
  "use"(%1) : (memref<32xf32>) -> ()
  memref.dealloc %1 : memref<32xf32>
  return
}
// CHECK: bufferization.clone
// CHECK: memref.dealloc
// CHECK: memref.dealloc

// -----

// Verify SimplifyClones skips clones with multiple deallocations.
// CHECK-LABEL: @clone_multiple_dealloc_of_source
func.func @clone_multiple_dealloc_of_source(%arg0: memref<?xf32>) -> memref<?xf32> {
  %0 = bufferization.clone %arg0 : memref<?xf32> to memref<?xf32>
  "if_else"() ({
    memref.dealloc %arg0 : memref<?xf32>
    }, {
    memref.dealloc %arg0 : memref<?xf32>
    }) : () -> ()
  return %0 : memref<?xf32>
}
// CHECK-SAME: %[[ARG:.*]]: memref<?xf32>
// CHECK-NEXT: %[[RES:.*]] = bufferization.clone %[[ARG]]
// CHECK: memref.dealloc %[[ARG]]
// CHECK: memref.dealloc %[[ARG]]
// CHECK: return %[[RES]]

// -----

// CHECK-LABEL: @clone_multiple_dealloc_of_clone
// CHECK-SAME: %[[ARG:.*]]: memref<?xf32>
func.func @clone_multiple_dealloc_of_clone(%arg0: memref<?xf32>) -> memref<?xf32> {
  // CHECK-NEXT: %[[CLONE:.*]] = bufferization.clone %[[ARG]]
  // CHECK: memref.dealloc %[[CLONE]]
  // CHECK: memref.dealloc %[[CLONE]]
  // CHECK: return %[[ARG]]
  %0 = bufferization.clone %arg0 : memref<?xf32> to memref<?xf32>
  "use"(%0) : (memref<?xf32>) -> ()
  "if_else"() ({
    memref.dealloc %0 : memref<?xf32>
    }, {
    memref.dealloc %0 : memref<?xf32>
    }) : () -> ()
  return %arg0 : memref<?xf32>
}

// -----

// Verify SimplifyClones skips clones followed by realloc.
// CHECK-LABEL: @clone_and_realloc
func.func @clone_and_realloc(%arg0: memref<?xf32>) {
  %0 = bufferization.clone %arg0 : memref<?xf32> to memref<32xf32>
  "use"(%0) : (memref<32xf32>) -> ()
  %1 = memref.realloc %0 : memref<32xf32> to memref<64xf32>
  memref.dealloc %1 : memref<64xf32>
  return
}
// CHECK-SAME: %[[ARG:.*]]: memref<?xf32>
// CHECK-NOT: %cast = memref.cast %[[ARG]]

// -----

// CHECK-LABEL: func @tensor_cast_to_memref
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<4x6x16x32xi8>
func.func @tensor_cast_to_memref(%arg0 : tensor<4x6x16x32xi8>) ->
  memref<?x?x16x32xi8> {
  %0 = tensor.cast %arg0 : tensor<4x6x16x32xi8> to tensor<?x?x16x32xi8>
  %1 = bufferization.to_memref %0 : memref<?x?x16x32xi8>
  return %1 : memref<?x?x16x32xi8>
}
// CHECK:   %[[M:.+]] = bufferization.to_memref %[[ARG0]] : memref<4x6x16x32xi8>
// CHECK:   %[[M1:.+]] = memref.cast %[[M]]
// CHECK-SAME: memref<4x6x16x32xi8> to memref<?x?x16x32xi8>
// CHECK:   return %[[M1]] : memref<?x?x16x32xi8>

// -----

// Folding of memref.load(to_memref(%v, %idxs)) -> tensor.extract(%v, %idx)
// CHECK-LABEL: func @load_from_buffer_cast(
func.func @load_from_buffer_cast(%arg0: index, %arg1: index,
                            %arg2: tensor<?x?xf32>) -> f32 {
  %0 = bufferization.to_memref %arg2 : memref<?x?xf32>
  %1 = memref.load %0[%arg0, %arg1] : memref<?x?xf32>
  return %1 : f32
}
//  CHECK-SAME: %[[IDX0:[0-9a-z]+]]: index, %[[IDX1:[0-9a-z]+]]: index
//  CHECK-SAME: %[[TENSOR:[0-9a-z]+]]: tensor<?x?xf32>
//       CHECK: %[[RES:.*]] = tensor.extract %[[TENSOR]][%[[IDX0]], %[[IDX1]]]
//   CHECK-NOT: memref.load
//       CHECK: return %[[RES]] : f32


// -----

func.func @alloc_tensor_canonicalize() -> (tensor<4x5x?xf32>) {
  %c6 = arith.constant 6 : index
  %0 = bufferization.alloc_tensor(%c6) : tensor<4x5x?xf32>
  return %0 : tensor<4x5x?xf32>
}
// CHECK: func @alloc_tensor_canonicalize
// CHECK:   %[[T0:.+]] = bufferization.alloc_tensor() : tensor<4x5x6xf32>
// CHECK:   %[[T1:.+]] = tensor.cast %[[T0]] : tensor<4x5x6xf32> to tensor<4x5x?xf32>
// CHECK:   return %[[T1]]

// -----

func.func @dealloc_canonicalize_clone_removal(%arg0: memref<?xindex>) -> memref<*xf32> {
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%c1) : memref<?xf32>
  %1 = memref.reshape %0(%arg0) : (memref<?xf32>, memref<?xindex>) -> memref<*xf32>
  %2 = bufferization.clone %1 : memref<*xf32> to memref<*xf32>
  memref.dealloc %0 : memref<?xf32>
  return %2 : memref<*xf32>
}
// CHECK-LABEL: @dealloc_canonicalize_clone_removal
//   CHECK-NOT:   bufferization.clone
//   CHECK-NOT:   memref.dealloc
//       CHECK:   return {{.*}}

// -----

func.func @dealloc_canonicalize_duplicates(%arg0: memref<2xi32>, %arg1: i1, %arg2: i1, %arg3: memref<2xi32>, %arg4: memref<2xi32>, %arg5: memref<2xi32>) -> (i1, i1, i1) {
  %0:3 = bufferization.dealloc (%arg4, %arg0, %arg0 : memref<2xi32>, memref<2xi32>, memref<2xi32>) if (%arg1, %arg1, %arg1) retain (%arg3, %arg5, %arg3 : memref<2xi32>, memref<2xi32>, memref<2xi32>)
  bufferization.dealloc (%arg0, %arg0 : memref<2xi32>, memref<2xi32>) if (%arg1, %arg2)
  return %0#0, %0#1, %0#2 : i1, i1, i1
}

// CHECK-LABEL: func @dealloc_canonicalize_duplicates
//  CHECK-SAME:  ([[ARG0:%.+]]: memref<2xi32>, [[ARG1:%.+]]: i1, [[ARG2:%.+]]: i1, [[ARG3:%.+]]: memref<2xi32>, [[ARG4:%.+]]: memref<2xi32>, [[ARG5:%.+]]: memref<2xi32>)
//  CHECK-NEXT:   [[V0:%.+]]:2 = bufferization.dealloc ([[ARG4]], [[ARG0]] : memref<2xi32>, memref<2xi32>) if ([[ARG1]], [[ARG1]]) retain ([[ARG3]], [[ARG5]] : memref<2xi32>, memref<2xi32>)
//  CHECK-NEXT:   [[NEW_COND:%.+]] = arith.ori [[ARG1]], [[ARG2]] : i1
//  CHECK-NEXT:   bufferization.dealloc ([[ARG0]] : memref<2xi32>) if ([[NEW_COND]])
//  CHECK-NEXT:   return [[V0]]#0, [[V0]]#1, [[V0]]#0 :

// -----

func.func @dealloc_erase_empty(%arg0: memref<2xi32>, %arg1: i1, %arg2: memref<2xi32>) -> i1 {
  bufferization.dealloc
  %0 = bufferization.dealloc retain (%arg0 : memref<2xi32>)
  return %0 : i1
}

// CHECK-LABEL: func @dealloc_erase_empty
//  CHECK-SAME: ([[ARG0:%.+]]: memref<2xi32>, [[ARG1:%.+]]: i1, [[ARG2:%.+]]: memref<2xi32>)
//  CHECK-NEXT: [[FALSE:%.+]] = arith.constant false
//  CHECK-NEXT: return [[FALSE]] :

// -----

func.func @dealloc_always_false_condition(%arg0: memref<2xi32>, %arg1: memref<2xi32>, %arg2: i1) {
  %false = arith.constant false
  bufferization.dealloc (%arg0, %arg1 : memref<2xi32>, memref<2xi32>) if (%false, %arg2)
  return
}

// CHECK-LABEL: func @dealloc_always_false_condition
//  CHECK-SAME: ([[ARG0:%.+]]: memref<2xi32>, [[ARG1:%.+]]: memref<2xi32>, [[ARG2:%.+]]: i1)
//  CHECK-NEXT: bufferization.dealloc ([[ARG1]] : {{.*}}) if ([[ARG2]])
//  CHECK-NEXT: return

// -----

func.func @dealloc_base_memref_extract_of_alloc(%arg0: memref<2xi32>, %arg1: i1, %arg2: i1, %arg3: memref<2xi32>) -> memref<2xi32> {
  %alloc = memref.alloc() : memref<2xi32>
  %base0, %size0, %stride0, %offset0 = memref.extract_strided_metadata %alloc : memref<2xi32> -> memref<i32>, index, index, index
  %base1, %size1, %stride1, %offset1 = memref.extract_strided_metadata %arg3 : memref<2xi32> -> memref<i32>, index, index, index
  bufferization.dealloc (%base0, %arg0, %base1 : memref<i32>, memref<2xi32>, memref<i32>) if (%arg1, %arg2, %arg2)
  return %alloc : memref<2xi32>
}

// CHECK-LABEL: func @dealloc_base_memref_extract_of_alloc
//  CHECK-SAME: ([[ARG0:%.+]]: memref<2xi32>, [[ARG1:%.+]]: i1, [[ARG2:%.+]]: i1, [[ARG3:%.+]]: memref<2xi32>)
//  CHECK-NEXT: [[ALLOC:%.+]] = memref.alloc() : memref<2xi32>
//  CHECK-NEXT: [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[ARG3]] :
//  CHECK-NEXT: bufferization.dealloc ([[ALLOC]], [[ARG0]], [[BASE]] : memref<2xi32>, memref<2xi32>, memref<i32>) if ([[ARG1]], [[ARG2]], [[ARG2]])
//  CHECK-NEXT: return

// -----

func.func @dealloc_base_memref_extract_of_alloc(%arg0: memref<2xi32>) {
  %true = arith.constant true
  %alloc = memref.alloc() : memref<2xi32>
  bufferization.dealloc (%alloc, %arg0 : memref<2xi32>, memref<2xi32>) if (%true, %true)
  return
}

// CHECK-LABEL: func @dealloc_base_memref_extract_of_alloc
//  CHECK-SAME:([[ARG0:%.+]]: memref<2xi32>)
//   CHECK-NOT: memref.alloc(
//       CHECK: bufferization.dealloc ([[ARG0]] : memref<2xi32>) if (%true
