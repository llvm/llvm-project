// RUN: mlir-opt %s -test-linalg-hoisting=test-hoist-view-allocs | FileCheck %s

// CHECK-LABEL: func @hoist(
//  CHECK-SAME:   %[[VAL:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[LB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[UB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[STEP:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[CMP:[a-zA-Z0-9]*]]: i1
func @hoist(%val: index, %lb : index, %ub : index, %step: index, %cmp: i1) {
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
