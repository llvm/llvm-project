// RUN: mlir-opt %s -allow-unregistered-dialect -test-decompose-affine-ops -split-input-file -loop-invariant-code-motion -cse | FileCheck %s

// CHECK-DAG: #[[$c42:.*]] = affine_map<() -> (42)>
// CHECK-DAG: #[[$div32mod4:.*]] = affine_map<()[s0] -> ((s0 floordiv 32) mod 4)>
// CHECK-DAG: #[[$add:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>

// CHECK-LABEL:  func.func @simple_test_1
//  CHECK-SAME:  %[[I0:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[I1:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[I2:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[LB:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[UB:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[STEP:[0-9a-zA-Z]+]]: index
func.func @simple_test_1(%0: index, %1: index, %2: index, %lb: index, %ub: index, %step: index) {
  // CHECK: %[[c42:.*]] = affine.apply #[[$c42]]()
  // CHECK: %[[R1:.*]] = affine.apply #[[$div32mod4]]()[%[[I1]]]
  // CHECK: %[[a:.*]] = affine.apply #[[$add]]()[%[[c42]], %[[R1]]]
  %a = affine.apply affine_map<(d0) -> ((d0 floordiv 32) mod 4 + 42)>(%1)

  // CHECK:     "some_side_effecting_consumer"(%[[a]]) : (index) -> ()
  "some_side_effecting_consumer"(%a) : (index) -> ()
  return
}

// -----

// CHECK-DAG: #[[$c42:.*]] = affine_map<() -> (42)>
// CHECK-DAG: #[[$id:.*]] = affine_map<()[s0] -> (s0)>
// CHECK-DAG: #[[$add:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-DAG: #[[$div32div4timesm4:.*]] = affine_map<()[s0] -> (((s0 floordiv 32) floordiv 4) * -4)>
// CHECK-DAG: #[[$div32:.*]] = affine_map<()[s0] -> (s0 floordiv 32)>

// CHECK-LABEL:  func.func @simple_test_2
//  CHECK-SAME:  %[[I0:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[I1:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[I2:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[LB:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[UB:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[STEP:[0-9a-zA-Z]+]]: index
func.func @simple_test_2(%0: index, %1: index, %2: index, %lb: index, %ub: index, %step: index) {
  // CHECK: %[[c42:.*]] = affine.apply #[[$c42]]()
  // CHECK: scf.for %[[i:.*]] =
  scf.for %i = %lb to %ub step %step {
    // CHECK:   %[[R1:.*]] = affine.apply #[[$id]]()[%[[i]]]
    // CHECK:   %[[R2:.*]] = affine.apply #[[$add]]()[%[[c42]], %[[R1]]]
    // CHECK:   scf.for %[[j:.*]] =
    scf.for %j = %lb to %ub step %step {
      // CHECK:     %[[R3:.*]] = affine.apply #[[$div32div4timesm4]]()[%[[j]]]
      // CHECK:     %[[R4:.*]] = affine.apply #[[$add]]()[%[[R2]], %[[R3]]]
      // CHECK:     %[[R5:.*]] = affine.apply #[[$div32]]()[%[[j]]]
      // CHECK:      %[[a:.*]] = affine.apply #[[$add]]()[%[[R4]], %[[R5]]]
      %a = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 32) mod 4 + s0 + 42)>(%j)[%i]

      // CHECK:     "some_side_effecting_consumer"(%[[a]]) : (index) -> ()
      "some_side_effecting_consumer"(%a) : (index) -> ()
    }
  }
  return
}

// -----

// CHECK-DAG: #[[$div4:.*]] = affine_map<()[s0] -> (s0 floordiv 4)>
// CHECK-DAG: #[[$times32:.*]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-DAG: #[[$times16:.*]] = affine_map<()[s0] -> (s0 * 16)>
// CHECK-DAG: #[[$add:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-DAG: #[[$div4timesm32:.*]] = affine_map<()[s0] -> ((s0 floordiv 4) * -32)>
// CHECK-DAG: #[[$times8:.*]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-DAG: #[[$id:.*]] = affine_map<()[s0] -> (s0)>
// CHECK-DAG: #[[$div32div4timesm4:.*]] = affine_map<()[s0] -> (((s0 floordiv 32) floordiv 4) * -4)>
// CHECK-DAG: #[[$div32:.*]] = affine_map<()[s0] -> (s0 floordiv 32)>

// CHECK-LABEL:  func.func @larger_test
//  CHECK-SAME:  %[[I0:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[I1:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[I2:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[LB:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[UB:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[STEP:[0-9a-zA-Z]+]]: index
func.func @larger_test(%0: index, %1: index, %2: index, %lb: index, %ub: index, %step: index) {
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index

    //      CHECK: %[[R0:.*]] = affine.apply #[[$div4]]()[%[[I0]]]
    // CHECK-NEXT: %[[R1:.*]] = affine.apply #[[$times16]]()[%[[I1]]]
    // CHECK-NEXT: %[[R2:.*]] = affine.apply #[[$add]]()[%[[R0]], %[[R1]]]
    // CHECK-NEXT: %[[R3:.*]] = affine.apply #[[$times32]]()[%[[I2]]]

    // I1 * 16 + I2 * 32 + I0 floordiv 4
    // CHECK-NEXT: %[[b:.*]] = affine.apply #[[$add]]()[%[[R2]], %[[R3]]]

    // (I0 floordiv 4) * 32
    // CHECK-NEXT: %[[R5:.*]] = affine.apply #[[$div4timesm32]]()[%[[I0]]]
    // 8 * I0
    // CHECK-NEXT: %[[R6:.*]] = affine.apply #[[$times8]]()[%[[I0]]]
    // 8 * I0 + (I0 floordiv 4) * 32
    // CHECK-NEXT: %[[c:.*]] = affine.apply #[[$add]]()[%[[R5]], %[[R6]]]

    // CHECK-NEXT: scf.for %[[i:.*]] =
    scf.for %i = %lb to %ub step %step {
      // remainder from %a not hoisted above %i.
      // CHECK-NEXT: %[[R8:.*]] = affine.apply #[[$times32]]()[%[[i]]]
      // CHECK-NEXT: %[[a:.*]] = affine.apply #[[$add]]()[%[[b]], %[[R8]]]

      // CHECK-NEXT: scf.for %[[j:.*]] =
      scf.for %j = %lb to %ub step %step {
        // Gets hoisted partially to i and rest outermost.
        // The hoisted part is %b.
        %a = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 16 + s2 * 32 + s3 * 32 + s0 floordiv 4)>()[%0, %1, %2, %i]

        // Gets completely hoisted 
        %b = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%0, %1, %2]

        // Gets completely hoisted 
        %c = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
 
        // 32 * %j + %c remains here, the rest is hoisted.
        // CHECK-DAG: %[[R10:.*]] = affine.apply #[[$times32]]()[%[[j]]]
        // CHECK-DAG: %[[d:.*]] = affine.apply #[[$add]]()[%[[c]], %[[R10]]]
        %d = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1 * 32 - (s0 floordiv 4) * 32)>()[%0, %j]

        // CHECK-DAG: %[[idj:.*]] = affine.apply #[[$id]]()[%[[j]]]
        // CHECK-NEXT: scf.for %[[k:.*]] =
        scf.for %k = %lb to %ub step %step {
          // CHECK-NEXT: %[[idk:.*]] = affine.apply #[[$id]]()[%[[k]]]
          // CHECK-NEXT: %[[e:.*]] = affine.apply #[[$add]]()[%[[c]], %[[idk]]]
          %e = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8 - (s1 floordiv 4) * 32)>()[%k, %0]

          // CHECK-NEXT: %[[R15:.*]] = affine.apply #[[$div32div4timesm4]]()[%[[k]]]
          // CHECK-NEXT: %[[R16:.*]] = affine.apply #[[$add]]()[%[[idj]], %[[R15]]]
          // CHECK-NEXT: %[[R17:.*]] = affine.apply #[[$div32]]()[%[[k]]]
          // CHECK-NEXT: %[[f:.*]] = affine.apply #[[$add]]()[%[[R16]], %[[R17]]]
          %f = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 32) mod 4 + s0)>(%k)[%j]

          // CHECK-NEXT: %[[g:.*]] = affine.apply #[[$add]]()[%[[b]], %[[idk]]]
          %g = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s2 * 16 + s3 * 32 + s1 floordiv 4)>()[%k, %0, %1, %2]
          
          // CHECK-NEXT: "some_side_effecting_consumer"(%[[a]]) : (index) -> ()
          "some_side_effecting_consumer"(%a) : (index) -> ()
          // CHECK-NEXT: "some_side_effecting_consumer"(%[[b]]) : (index) -> ()
          "some_side_effecting_consumer"(%b) : (index) -> ()
          // CHECK-NEXT: "some_side_effecting_consumer"(%[[c]]) : (index) -> ()
          "some_side_effecting_consumer"(%c) : (index) -> ()
          // CHECK-NEXT: "some_side_effecting_consumer"(%[[d]]) : (index) -> ()
          "some_side_effecting_consumer"(%d) : (index) -> ()
          // CHECK-NEXT: "some_side_effecting_consumer"(%[[e]]) : (index) -> ()
          "some_side_effecting_consumer"(%e) : (index) -> ()
          // CHECK-NEXT: "some_side_effecting_consumer"(%[[f]]) : (index) -> ()
          "some_side_effecting_consumer"(%f) : (index) -> ()
          // CHECK-NEXT: "some_side_effecting_consumer"(%[[g]]) : (index) -> ()
          "some_side_effecting_consumer"(%g) : (index) -> ()
        }
    }
  }   
  return
}
