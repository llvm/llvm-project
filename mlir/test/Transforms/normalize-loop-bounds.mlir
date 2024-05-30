// RUN: mlir-opt %s  -split-input-file -normalize-loop-bounds -verify-diagnostics | FileCheck %s

// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-LABEL: func.func @for_lowerbound_static
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       scf.for %[[ARG:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK-NEXT:    affine.apply #[[$MAP]](%[[ARG]])
module {
  func.func @for_lowerbound_static() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    scf.for %arg0 = %c2 to %c8 step %c1 {
    }
    return
  }
}

// -----

// CHECK:       #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: func.func @for_lowerbound_dynamic
// CHECK-SAME:  %[[ARG0:.+]]: index
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[UB:.+]] = arith.subi %[[C8]], %[[ARG0]] : index
// CHECK:       scf.for %[[ARG:.+]] = %[[C0]] to %[[UB]] step %[[C1]]
// CHECK-NEXT:    affine.apply #[[$MAP]](%[[ARG]], %[[ARG0]])
module {
  func.func @for_lowerbound_dynamic(%lb: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %arg0 = %lb to %c8 step %c1 {
    }
    return
  }
}

// -----

// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL: func.func @for_step_static
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK:       scf.for %[[ARG:.+]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK-NEXT:    affine.apply #[[$MAP]](%[[ARG]])
module {
  func.func @for_step_static() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    scf.for %arg0 = %c0 to %c8 step %c2 {
    }
    return
  }
}

// -----

// CHECK:       #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0 * d1)>
// CHECK-LABEL: func.func @for_step_dynamic
// CHECK-SAME:  %[[ARG0:.+]]: index
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[UB:.+]] = arith.ceildivsi %[[C8]], %[[ARG0]] : index
// CHECK:       scf.for %[[ARG:.+]] = %[[C0]] to %[[UB]] step %[[C1]]
// CHECK-NEXT:    affine.apply #[[$MAP]](%[[ARG]], %[[ARG0]])
module {
  func.func @for_step_dynamic(%step: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    scf.for %arg0 = %c0 to %c8 step %step {
    }
    return
  }
}

// -----

// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 4 + 1)>
// CHECK-LABEL: func.func @for_lowerbound_and_step_static
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
// CHECK:       scf.for %[[ARG:.+]] = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK-NEXT:    affine.apply #[[$MAP]](%[[ARG]])
module {
  func.func @for_lowerbound_and_step_static() {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c13 = arith.constant 13 : index
    scf.for %arg0 = %c1 to %c13 step %c4 {
    }
    return
  }
}

// -----

// CHECK:       #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0 * d1 + d2)>
// CHECK-LABEL: func.func @for_lowerbound_and_step_dynamic
// CHECK-SAME:  %[[LB:.+]]: index, %[[STEP:.+]]: index
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C13:.+]] = arith.constant 13 : index
// CHECK-DAG:   %[[SUB:.+]] = arith.subi %[[C13]], %[[LB]] : index
// CHECK-DAG:   %[[UB:.+]] = arith.ceildivsi %[[SUB]], %[[STEP]] : index
// CHECK:       scf.for %[[ARG:.+]] = %[[C0]] to %[[UB]] step %[[C1]]
// CHECK-NEXT:    affine.apply #[[$MAP]](%[[ARG]], %[[STEP]], %[[LB]])
module {
  func.func @for_lowerbound_and_step_dynamic(%lb: index, %step: index) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c13 = arith.constant 13 : index
    scf.for %arg0 = %lb to %c13 step %step {
    }
    return
  }
}

// -----

// CHECK-DAG:   #[[$MAP0:.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-DAG:   #[[$MAP1:.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-LABEL: func.func @forall_lowerbound_static
// CHECK:       scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) in (6, 12)
// CHECK-DAG:     affine.apply #[[$MAP1]](%[[ARG0]])
// CHECK-DAG:     affine.apply #[[$MAP0]](%[[ARG1]])
module {
  func.func @forall_lowerbound_static() {
    scf.forall (%arg2, %arg3) = (2, 4) to (8, 16) step (1, 1) {
    }
    return
  }
}

// -----

// CHECK-DAG:   #[[$MAP0:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: func.func @forall_lowerbound_dynamic
// CHECK-SAME:  %[[LB0:.+]]: index, %[[LB1:.+]]: index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[UB0:.+]] = arith.subi %[[C8]], %[[LB0]] : index
// CHECK-DAG:   %[[UB1:.+]] = arith.subi %[[C16]], %[[LB1]] : index
// CHECK:       scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) in (%[[UB0]], %[[UB1]])
// CHECK-DAG:     affine.apply #[[$MAP0]](%[[ARG0]], %[[LB0]])
// CHECK-DAG:     affine.apply #[[$MAP0]](%[[ARG1]], %[[LB1]])
module {
  func.func @forall_lowerbound_dynamic(%lb0: index, %lb1: index) {
    scf.forall (%arg2, %arg3) = (%lb0, %lb1) to (8, 16) step (1, 1) {
    }
    return
  }
}

// -----

// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-LABEL: func.func @forall_step_static
// CHECK:       scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) in (1, 2)
// CHECK-DAG:     affine.apply #[[$MAP]](%[[ARG0]])
// CHECK-DAG:     affine.apply #[[$MAP]](%[[ARG1]])
module {
  func.func @forall_step_static() {
    scf.forall (%arg2, %arg3) = (0, 0) to (8, 16) step (8, 8) {
    }
    return
  }
}

// -----

// CHECK:       #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0 * d1)>
// CHECK-LABEL: func.func @forall_step_dynamic
// CHECK-SAME:  %[[STEP0:.+]]: index, %[[STEP1:.+]]: index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[UB0:.+]] = arith.ceildivsi %[[C8]], %[[STEP0]] : index
// CHECK-DAG:   %[[UB1:.+]] = arith.ceildivsi %[[C16]], %[[STEP1]] : index
// CHECK:       scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) in (%[[UB0]], %[[UB1]])
// CHECK-DAG:     affine.apply #[[$MAP]](%[[ARG0]], %[[STEP0]])
// CHECK-DAG:     affine.apply #[[$MAP]](%[[ARG1]], %[[STEP1]])
module {
  func.func @forall_step_dynamic(%step0: index, %step1: index) {
    scf.forall (%arg2, %arg3) = (0, 0) to (8, 16) step (%step0, %step1) {
    }
    return
  }
}

// -----

// CHECK-DAG:   #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 4 + 4)>
// CHECK-DAG:   #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * 2 + 2)>
// CHECK-LABEL: func.func @forall_lowerbound_and_step_static
// CHECK:       scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) in (3, 3)
// CHECK-DAG:     affine.apply #[[$MAP1]](%[[ARG0]])
// CHECK-DAG:     affine.apply #[[$MAP0]](%[[ARG1]])
module {
  func.func @forall_lowerbound_and_step_static() {
    scf.forall (%arg2, %arg3) = (2, 4) to (8, 16) step (2, 4) {
    }
    return
  }
}

// -----

// CHECK-DAG:   #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0 * d1 + d2)>
// CHECK-LABEL: func.func @forall_lowerbound_and_step_dynamic
// CHECK-SAME:  %[[LB0:.+]]: index, %[[LB1:.+]]: index, %[[STEP0:.+]]: index, %[[STEP1:.+]]: index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[SUB0:.+]] = arith.subi %[[C8]], %[[LB0]] : index
// CHECK-DAG:   %[[SUB1:.+]] = arith.subi %[[C16]], %[[LB1]] : index
// CHECK-DAG:   %[[UB0:.+]] = arith.ceildivsi %[[SUB0]], %[[STEP0]] : index
// CHECK-DAG:   %[[UB1:.+]] = arith.ceildivsi %[[SUB1]], %[[STEP1]] : index
// CHECK:       scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) in (%[[UB0]], %[[UB1]])
// CHECK-DAG:     affine.apply #[[$MAP]](%[[ARG0]], %[[STEP0]], %[[LB0]])
// CHECK-DAG:     affine.apply #[[$MAP]](%[[ARG1]], %[[STEP1]], %[[LB1]])
module {
  func.func @forall_lowerbound_and_step_dynamic(%lb0: index, %lb1: index, %step0: index, %step1: index) {
    scf.forall (%arg2, %arg3) = (%lb0, %lb1) to (8, 16) step (%step0, %step1) {
    }
    return
  }
}

// -----

// CHECK-DAG:   #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 4 + 2)>
// CHECK-LABEL: func.func @forall_with_shared_outs_static
// CHECK-SAME:  %[[OUT:.+]]: tensor<200x100xf32>
// CHECK:       scf.forall (%[[ARG0:.+]]) in (2) shared_outs(%{{.+}} = %[[OUT]])
// CHECK-DAG:     affine.apply #[[$MAP]](%[[ARG0]])
module {
  func.func @forall_with_shared_outs_static(%out: tensor<200x100xf32>) {
    scf.forall (%arg0) = (2) to (8) step (4) shared_outs (%o = %out) -> tensor<200x100xf32> {
    }
    return
  }
}

// -----

// CHECK-DAG:   #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0 * d1 + d2)>
// CHECK-LABEL: func.func @forall_with_shared_outs_dynamic
// CHECK-SAME:  %[[LB:.+]]: index, %[[STEP:.+]]: index, %[[OUT:.+]]: tensor<200x100xf32>
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[SUB:.+]] = arith.subi %[[C8]], %[[LB]] : index
// CHECK-DAG:   %[[UB:.+]] = arith.ceildivsi %[[SUB]], %[[STEP]] : index
// CHECK:       scf.forall (%[[ARG:.+]]) in (%[[UB]]) shared_outs(%{{.+}} = %[[OUT]])
// CHECK-DAG:     affine.apply #[[$MAP]](%[[ARG]], %[[STEP]], %[[LB]])
module {
  func.func @forall_with_shared_outs_dynamic(%lb: index, %step: index, %out: tensor<200x100xf32>) {
    scf.forall (%arg0) = (%lb) to (8) step (%step) shared_outs (%o = %out) -> tensor<200x100xf32> {
    }
    return
  }
}
