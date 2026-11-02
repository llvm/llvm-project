// RUN: mlir-opt  -transform-interpreter -canonicalize --split-input-file %s | FileCheck %s

// Test hoisting of vector.extract/vector.broadcast pairs

// CHECK-LABEL:  func.func @hoist_vector_broadcasts
//       CHECK-SAME: (%{{.+}}: index, %{{.+}}: index, %{{.+}}: index, %[[VEC:.+]]: vector<3x4xf32>) -> vector<3x4xf32> {
//       CHECK:        %[[EXTRACT:.+]] = vector.extract %[[VEC]][0] : vector<4xf32> from vector<3x4xf32>
//       CHECK-NEXT:   %[[LOOP:.+]] = scf.for {{.*}} {
//       CHECK-NEXT:     %[[USE:.+]] = "test.some_use"({{.*}}) : (vector<4xf32>) -> vector<4xf32>
//       CHECK-NEXT:     scf.yield %[[USE]] : vector<4xf32>
//       CHECK-NEXT:   }
//       CHECK-NEXT:   %[[BCAST:.+]] = vector.broadcast %[[LOOP]] : vector<4xf32> to vector<3x4xf32>
//       CHECK-NEXT:   return %[[BCAST]] : vector<3x4xf32>

func.func @hoist_vector_broadcasts(%lb : index, %ub : index, %step : index, %vec : vector<3x4xf32>) -> vector<3x4xf32> {
  %bcast_vec = scf.for %arg0 = %lb to %ub step %step iter_args(%iarg = %vec) -> vector<3x4xf32> {
    %extract = vector.extract %iarg[0] : vector<4xf32> from vector<3x4xf32>
    %use = "test.some_use"(%extract) : (vector<4xf32>) -> vector<4xf32>
    %broadcast = vector.broadcast %use : vector<4xf32> to vector<3x4xf32>
    scf.yield %broadcast : vector<3x4xf32>
  }
  return %bcast_vec : vector<3x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.hoist_redundant_vector_broadcasts %0
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Test hoisting of vector.extract/vector.broadcast pairs with dynamic position

// CHECK-LABEL:  func.func @hoist_vector_broadcasts
//       CHECK-SAME: (%{{.+}}: index, %{{.+}}: index, %{{.+}}: index, %[[VEC:.+]]: vector<3x4xf32>, %[[POS:.+]]: index) -> vector<3x4xf32> {
//       CHECK:        %[[EXTRACT:.+]] = vector.extract %[[VEC]][%[[POS]]] : vector<4xf32> from vector<3x4xf32>
//       CHECK-NEXT:   %[[LOOP:.+]] = scf.for {{.*}} {
//       CHECK-NEXT:     %[[USE:.+]] = "test.some_use"({{.*}}) : (vector<4xf32>) -> vector<4xf32>
//       CHECK-NEXT:     scf.yield %[[USE]] : vector<4xf32>
//       CHECK-NEXT:   }
//       CHECK-NEXT:   %[[BCAST:.+]] = vector.broadcast %[[LOOP]] : vector<4xf32> to vector<3x4xf32>
//       CHECK-NEXT:   return %[[BCAST]] : vector<3x4xf32>

func.func @hoist_vector_broadcasts_dynamic(%lb : index, %ub : index, %step : index, %vec : vector<3x4xf32>, %pos: index) -> vector<3x4xf32> {
  %bcast_vec = scf.for %arg0 = %lb to %ub step %step iter_args(%iarg = %vec) -> vector<3x4xf32> {
    %extract = vector.extract %iarg[%pos] : vector<4xf32> from vector<3x4xf32>
    %use = "test.some_use"(%extract) : (vector<4xf32>) -> vector<4xf32>
    %broadcast = vector.broadcast %use : vector<4xf32> to vector<3x4xf32>
    scf.yield %broadcast : vector<3x4xf32>
  }
  return %bcast_vec : vector<3x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.hoist_redundant_vector_broadcasts %0
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Test hoisting of vector.extract/vector.broadcast pairs with multiple iter_args

// CHECK-LABEL:  func.func @hoist_vector_broadcasts_multiple
//       CHECK-SAME: (%{{.+}}: index, %{{.+}}: index, %{{.+}}: index, %[[VEC1:.+]]: vector<3x4xf32>,
//       CHECK-SAME:  %[[VEC2:.+]]: vector<3x5xf32>) -> (vector<3x4xf32>, vector<3x5xf32>) {
//       CHECK-DAG:     %[[EXTRACT1:.+]] = vector.extract %[[VEC1]][0] : vector<4xf32> from vector<3x4xf32>
//       CHECK-DAG:     %[[EXTRACT2:.+]] = vector.extract %[[VEC2]][1] : vector<5xf32> from vector<3x5xf32>
//       CHECK-NEXT:    %[[LOOP:.+]]:2 = scf.for {{.*}} {
//       CHECK-DAG:       %[[USE1:.+]] = "test.some_use1"({{.*}}) : (vector<4xf32>) -> vector<4xf32>
//       CHECK-DAG:       %[[USE2:.+]] = "test.some_use2"({{.*}}) : (vector<5xf32>) -> vector<5xf32>
//       CHECK-NEXT:      scf.yield %[[USE1]], %[[USE2]]  : vector<4xf32>, vector<5xf32>
//       CHECK-NEXT:    }
//       CHECK-DAG:     %[[BCAST1:.+]] = vector.broadcast %[[LOOP]]#0 : vector<4xf32> to vector<3x4xf32>
//       CHECK-DAG:     %[[BCAST2:.+]] = vector.broadcast %[[LOOP]]#1 : vector<5xf32> to vector<3x5xf32>
//       CHECK-NEXT:    return %[[BCAST1]], %[[BCAST2]] : vector<3x4xf32>, vector<3x5xf32>

func.func @hoist_vector_broadcasts_multiple(%lb : index, %ub : index, %step : index, %vec1 : vector<3x4xf32>, %vec2 : vector<3x5xf32>) ->  (vector<3x4xf32>, vector<3x5xf32>) {
  %bcast_vec:2 = scf.for %arg0 = %lb to %ub step %step iter_args(%iarg = %vec1, %iarg2 = %vec2) -> (vector<3x4xf32>, vector<3x5xf32>) {
    %extract1 = vector.extract %iarg[0] : vector<4xf32> from vector<3x4xf32>
    %extract2 = vector.extract %iarg2[1] : vector<5xf32> from vector<3x5xf32>
    %use1 = "test.some_use1"(%extract1) : (vector<4xf32>) -> vector<4xf32>
    %use2 = "test.some_use2"(%extract2) : (vector<5xf32>) -> vector<5xf32>
    %broadcast1 = vector.broadcast %use1 : vector<4xf32> to vector<3x4xf32>
    %broadcast2 = vector.broadcast %use2 : vector<5xf32> to vector<3x5xf32>
    scf.yield %broadcast1, %broadcast2 : vector<3x4xf32>,vector<3x5xf32>
  }
  return %bcast_vec#0, %bcast_vec#1 :  vector<3x4xf32>, vector<3x5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.hoist_redundant_vector_broadcasts %0
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
