// RUN: mlir-opt %s -test-linalg-hoisting=test-hoist-view-allocs -allow-unregistered-dialect | FileCheck %s
// RUN: mlir-opt %s -test-linalg-hoisting=test-hoist-redundant-transfers -allow-unregistered-dialect | FileCheck %s --check-prefix=VECTOR_TRANSFERS

// CHECK-LABEL: func @hoist_allocs(
//  CHECK-SAME:   %[[VAL:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[LB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[UB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[STEP:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[CMP:[a-zA-Z0-9]*]]: i1
func @hoist_allocs(%val: index, %lb : index, %ub : index, %step: index, %cmp: i1) {
//   CHECK-DAG:   alloca(%[[VAL]]) : memref<?xi8>
//   CHECK-DAG: %[[A0:.*]] = alloc(%[[VAL]]) : memref<?xi8>
//       CHECK: scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] {
//       CHECK:   alloca(%[[I]]) : memref<?xi8>
//       CHECK:   %[[A1:.*]] = alloc(%[[I]]) : memref<?xi8>
//       CHECK:   scf.for %[[J:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] {
//   CHECK-DAG:     alloca(%[[J]]) : memref<?xi8>
//   CHECK-DAG:     %[[A2:.*]] = alloc(%[[J]]) : memref<?xi8>
//       CHECK:     scf.for %[[K:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] {
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      scf.for %k = %lb to %ub step %step {
        // Hoist allocs / deallocs outermost, keep view/subview below k.
        %sa0 = alloca(%val) : memref<? x i8>
        %a0 = alloc(%val) : memref<? x i8>
//       CHECK:       std.view %[[A0]][%[[LB]]][] : memref<?xi8> to memref<16xf32>
//       CHECK:       subview %[[A0]][0] [42] [1]  : memref<?xi8> to memref<42xi8>
        %v0 = view %a0[%lb][] : memref<? x i8> to memref<16 x f32>
        %sv0 = subview %a0[0][42][1] : memref<? x i8> to memref<42 x i8>
        dealloc %a0 : memref<? x i8>

        // Hoist below i.
        %sa1 = alloca(%i) : memref<? x i8>
        %a1 = alloc(%i) : memref<? x i8>
        dealloc %a1 : memref<? x i8>

        // Hoist below j.
        %sa2 = alloca(%j) : memref<? x i8>
        %a2 = alloc(%j) : memref<? x i8>
        dealloc %a2 : memref<? x i8>

        // Don't hoist since k innermost.
//       CHECK:       alloca(%[[K]]) : memref<?xi8>
//       CHECK:       %[[A3:.*]] = alloc(%[[K]]) : memref<?xi8>
//       CHECK:       dealloc %[[A3]] : memref<?xi8>
        %sa3 = alloca(%k) : memref<? x i8>
        %a3 = alloc(%k) : memref<? x i8>
        dealloc %a3 : memref<? x i8>

        // No hoisting due to control flow.
//       CHECK:       scf.if %[[CMP]] {
//       CHECK:         alloca(%[[VAL]]) : memref<?xi8>
//       CHECK:         %[[A4:.*]] = alloc(%[[VAL]]) : memref<?xi8>
//       CHECK:         dealloc %[[A4]] : memref<?xi8>
        scf.if %cmp {
          %sa4 = alloca(%val) : memref<? x i8>
          %a4 = alloc(%val) : memref<? x i8>
          dealloc %a4 : memref<? x i8>
        }

        // No hoisting due to load/store.
//       CHECK:       %[[SA5:.*]] = alloca(%[[VAL]]) : memref<?xi8>
//       CHECK:       %[[A5:.*]] = alloc(%[[VAL]]) : memref<?xi8>
//       CHECK:       load %[[A5]][%[[LB]]] : memref<?xi8>
//       CHECK:       store %{{.*}}, %[[SA5]][%[[LB]]] : memref<?xi8>
//       CHECK:       dealloc %[[A5]] : memref<?xi8>
        %sa5 = alloca(%val) : memref<? x i8>
        %a5 = alloc(%val) : memref<? x i8>
        %v5 = load %a5[%lb] : memref<? x i8>
        store %v5, %sa5[%lb] : memref<? x i8>
        dealloc %a5 : memref<? x i8>

      }
    }
  }
//       CHECK:     }
//       CHECK:     dealloc %[[A2]] : memref<?xi8>
//       CHECK:   }
//       CHECK:   dealloc %[[A1]] : memref<?xi8>
//       CHECK: }
//       CHECK: dealloc %[[A0]] : memref<?xi8>
  return
}

// VECTOR_TRANSFERS-LABEL: func @hoist_vector_transfer_pairs(
//  VECTOR_TRANSFERS-SAME:   %[[MEMREF0:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  VECTOR_TRANSFERS-SAME:   %[[MEMREF1:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  VECTOR_TRANSFERS-SAME:   %[[MEMREF2:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  VECTOR_TRANSFERS-SAME:   %[[MEMREF3:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  VECTOR_TRANSFERS-SAME:   %[[MEMREF4:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  VECTOR_TRANSFERS-SAME:   %[[MEMREF5:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  VECTOR_TRANSFERS-SAME:   %[[VAL:[a-zA-Z0-9]*]]: index,
//  VECTOR_TRANSFERS-SAME:   %[[LB:[a-zA-Z0-9]*]]: index,
//  VECTOR_TRANSFERS-SAME:   %[[UB:[a-zA-Z0-9]*]]: index,
//  VECTOR_TRANSFERS-SAME:   %[[STEP:[a-zA-Z0-9]*]]: index,
//  VECTOR_TRANSFERS-SAME:   %[[CMP:[a-zA-Z0-9]*]]: i1
func @hoist_vector_transfer_pairs(
    %memref0: memref<?x?xf32>, %memref1: memref<?x?xf32>, %memref2: memref<?x?xf32>,
    %memref3: memref<?x?xf32>, %memref4: memref<?x?xf32>, %memref5: memref<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index, %cmp: i1) {
  %c0 = constant 0 : index
  %cst = constant 0.0 : f32

// VECTOR_TRANSFERS: vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<1xf32>
// VECTOR_TRANSFERS: scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}}) -> (vector<1xf32>) {
// VECTOR_TRANSFERS:   vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<2xf32>
// VECTOR_TRANSFERS:   scf.for %[[J:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}}) -> (vector<1xf32>, vector<2xf32>) {
// VECTOR_TRANSFERS:     vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<3xf32>
// VECTOR_TRANSFERS:     vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<4xf32>
// VECTOR_TRANSFERS:     "some_crippling_use"(%[[MEMREF4]]) : (memref<?x?xf32>) -> ()
// VECTOR_TRANSFERS:     vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<5xf32>
// VECTOR_TRANSFERS:     "some_use"(%{{.*}}) : (vector<1xf32>) -> vector<1xf32>
// VECTOR_TRANSFERS:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// VECTOR_TRANSFERS:     "some_use"(%[[MEMREF2]]) : (memref<?x?xf32>) -> vector<3xf32>
// VECTOR_TRANSFERS:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// VECTOR_TRANSFERS:     "some_use"(%{{.*}}) : (vector<5xf32>) -> vector<5xf32>
// VECTOR_TRANSFERS:     vector.transfer_write %{{.*}} : vector<3xf32>, memref<?x?xf32>
// VECTOR_TRANSFERS:     vector.transfer_write %{{.*}} : vector<4xf32>, memref<?x?xf32>
// VECTOR_TRANSFERS:     vector.transfer_write %{{.*}} : vector<5xf32>, memref<?x?xf32>
// VECTOR_TRANSFERS:     "some_crippling_use"(%[[MEMREF3]]) : (memref<?x?xf32>) -> ()
// VECTOR_TRANSFERS:     scf.yield {{.*}} : vector<1xf32>, vector<2xf32>
// VECTOR_TRANSFERS:   }
// VECTOR_TRANSFERS:   vector.transfer_write %{{.*}} : vector<2xf32>, memref<?x?xf32>
// VECTOR_TRANSFERS:   scf.yield {{.*}} : vector<1xf32>
// VECTOR_TRANSFERS: }
// VECTOR_TRANSFERS: vector.transfer_write %{{.*}} : vector<1xf32>, memref<?x?xf32>
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      %r0 = vector.transfer_read %memref1[%c0, %c0], %cst: memref<?x?xf32>, vector<1xf32>
      %r1 = vector.transfer_read %memref0[%i, %i], %cst: memref<?x?xf32>, vector<2xf32>
      %r2 = vector.transfer_read %memref2[%c0, %c0], %cst: memref<?x?xf32>, vector<3xf32>
      %r3 = vector.transfer_read %memref3[%c0, %c0], %cst: memref<?x?xf32>, vector<4xf32>
      "some_crippling_use"(%memref4) : (memref<?x?xf32>) -> ()
      %r4 = vector.transfer_read %memref4[%c0, %c0], %cst: memref<?x?xf32>, vector<5xf32>
      %u0 = "some_use"(%r0) : (vector<1xf32>) -> vector<1xf32>
      %u1 = "some_use"(%r1) : (vector<2xf32>) -> vector<2xf32>
      %u2 = "some_use"(%memref2) : (memref<?x?xf32>) -> vector<3xf32>
      %u3 = "some_use"(%r3) : (vector<4xf32>) -> vector<4xf32>
      %u4 = "some_use"(%r4) : (vector<5xf32>) -> vector<5xf32>
      vector.transfer_write %u0, %memref1[%c0, %c0] : vector<1xf32>, memref<?x?xf32>
      vector.transfer_write %u1, %memref0[%i, %i] : vector<2xf32>, memref<?x?xf32>
      vector.transfer_write %u2, %memref2[%c0, %c0] : vector<3xf32>, memref<?x?xf32>
      vector.transfer_write %u3, %memref3[%c0, %c0] : vector<4xf32>, memref<?x?xf32>
      vector.transfer_write %u4, %memref4[%c0, %c0] : vector<5xf32>, memref<?x?xf32>
      "some_crippling_use"(%memref3) : (memref<?x?xf32>) -> ()
    }
  }
  return
}
