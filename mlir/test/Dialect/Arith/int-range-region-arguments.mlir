// RUN: mlir-opt %s -int-range-optimizations | FileCheck %s

func.func private @consume(i32)

// CHECK-LABEL: func @forall_shared
// CHECK: shared_outs(%[[OUT:.*]] =
// CHECK: %[[SELECT:.*]] = arith.select %{{.*}}, %[[OUT]],
// CHECK: %[[MERGE:.*]] = scf.if
// CHECK: scf.yield %[[SELECT]]
// CHECK: %[[ELEMENT:.*]] = tensor.extract %[[MERGE]]
// CHECK: call @consume(%[[ELEMENT]])
func.func @forall_shared(%input: tensor<1xi32>, %c: i1, %d: i1) -> tensor<1xi32> {
  %zero = arith.constant dense<0> : tensor<1xi32>
  %idx = arith.constant 0 : index
  %r = scf.forall (%iv) in (1) shared_outs(%out = %input) -> tensor<1xi32> {
    %s = arith.select %c, %out, %zero : tensor<1xi32>
    %v = scf.if %d -> tensor<1xi32> {
      scf.yield %s : tensor<1xi32>
    } else {
      scf.yield %zero : tensor<1xi32>
    }
    %elem = tensor.extract %v[%idx] : tensor<1xi32>
    func.call @consume(%elem) : (i32) -> ()
    scf.forall.in_parallel {
    }
  }
  return %r : tensor<1xi32>
}

// CHECK-LABEL: func @forall_result
// CHECK: %[[RESULT:.*]] = scf.forall
// CHECK: %[[SELECT:.*]] = arith.select %{{.*}}, %[[RESULT]],
// CHECK: %[[MERGE:.*]] = scf.if
// CHECK: scf.yield %[[SELECT]]
// CHECK: return %[[MERGE]]
func.func @forall_result(%input: tensor<1xi32>, %c: i1, %d: i1) -> tensor<1xi32> {
  %zero = arith.constant dense<0> : tensor<1xi32>
  %r = scf.forall (%iv) in (1) shared_outs(%out = %input) -> tensor<1xi32> {
    scf.forall.in_parallel {
    }
  }
  %s = arith.select %c, %r, %zero : tensor<1xi32>
  %v = scf.if %d -> tensor<1xi32> {
    scf.yield %s : tensor<1xi32>
  } else {
    scf.yield %zero : tensor<1xi32>
  }
  return %v : tensor<1xi32>
}
