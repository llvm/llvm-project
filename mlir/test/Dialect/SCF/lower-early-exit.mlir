// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(scf-lower-early-exit))' -split-input-file | FileCheck %s

// Scenario 1: `scf.break` directly in the target's entry block becomes
// `scf.yield`.

// CHECK-LABEL:   func.func @break_in_entry(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: f32) -> f32 {
// CHECK:           %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> f32 {
// CHECK:             scf.yield %[[ARG0]] : f32
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : f32
// CHECK:         }
func.func @break_in_entry(%v: f32) -> f32 {
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok: token):
    scf.break %tok, %v : f32
  }
  return %0 : f32
}

// -----

// Scenario 2: `scf.break` inside an `scf.if`, with trailing ops after the if.
// After lowering, no break remains, the if has both branches yielding f32,
// and the trailing ops live in the else branch.

// CHECK-LABEL:   func.func @break_in_if(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_]*]]: f32) -> f32 {
// CHECK:           %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> f32 {
// CHECK:             %[[IF_0:.*]] = scf.if %[[ARG0]] -> (f32) {
// CHECK:               scf.yield %[[ARG1]] : f32
// CHECK:             } else {
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[ARG2]], %[[ARG2]] : f32
// CHECK:               scf.yield %[[ADDF_0]] : f32
// CHECK:             }
// CHECK:             scf.yield %[[IF_0]] : f32
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : f32
// CHECK:         }
func.func @break_in_if(%cond: i1, %v1: f32, %v2: f32) -> f32 {
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok: token):
    scf.if %cond {
      scf.break %tok, %v1 : f32
    }
    %1 = arith.addf %v2, %v2 : f32
    scf.yield %1 : f32
  }
  return %0 : f32
}

// -----

// Scenario 3: `scf.break` inside an inner `scf.execute_region` targeting the
// outer one. The inner execute_region is augmented with extra results to
// thread the broken-out value and a flag, and a guarded re-break is inserted
// in the outer body.

// CHECK-LABEL:   func.func @break_through_execute_region(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_]*]]: f32) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant false
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant true
// CHECK:           %[[POISON_0:.*]] = ub.poison : f32
// CHECK:           %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> f32 {
// CHECK:             %[[EXECUTE_REGION_1:.*]]:3 = scf.execute_region -> (f32, i1, f32) {
// CHECK:               %[[IF_0:.*]]:3 = scf.if %[[ARG0]] -> (f32, i1, f32) {
// CHECK:                 scf.yield %[[POISON_0]], %[[CONSTANT_1]], %[[ARG1]] : f32, i1, f32
// CHECK:               } else {
// CHECK:                 "test.some_compute"() : () -> ()
// CHECK:                 scf.yield %[[ARG2]], %[[CONSTANT_0]], %[[POISON_0]] : f32, i1, f32
// CHECK:               }
// CHECK:               scf.yield %[[VAL_0:.*]]#0, %[[VAL_0]]#1, %[[VAL_0]]#2 : f32, i1, f32
// CHECK:             }
// CHECK:             %[[IF_1:.*]] = scf.if %[[VAL_1:.*]]#1 -> (f32) {
// CHECK:               scf.yield %[[VAL_1]]#2 : f32
// CHECK:             } else {
// CHECK:               %[[VAL_2:.*]] = "test.use"(%[[VAL_3:.*]]#0) : (f32) -> f32
// CHECK:               scf.yield %[[VAL_2]] : f32
// CHECK:             }
// CHECK:             scf.yield %[[IF_1]] : f32
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : f32
// CHECK:         }
func.func @break_through_execute_region(%c1: i1, %v1: f32, %v2: f32) -> f32 {
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok: token):
    %b = scf.execute_region -> f32 {
    ^bb0(%tok2: token):
      scf.if %c1 {
        scf.break %tok, %v1 : f32
      }
      "test.some_compute"() : () -> ()
      scf.yield %v2 : f32
    }
    %u = "test.use"(%b) : (f32) -> f32
    scf.yield %u : f32
  }
  return %0 : f32
}

// -----

// A pre-existing token-bearing execute_region with no break inside is left
// structurally similar (the body's yield remains the only terminator).

// CHECK-LABEL:   func.func @no_break_inside(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: f32) -> f32 {
// CHECK:           %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> f32 {
// CHECK:             scf.yield %[[ARG0]] : f32
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : f32
// CHECK:         }
func.func @no_break_inside(%v: f32) -> f32 {
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok: token):
    scf.yield %v : f32
  }
  return %0 : f32
}

// -----

// A break nested inside two `scf.if` ops (both forwarding through the target's
// yield) should be replaced with a yield after sinking trailing ops into the
// fall-through branches.

// CHECK-LABEL:   func.func @break_in_nested_if(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG3:[a-zA-Z0-9_]*]]: f32) -> f32 {
// CHECK:           %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> f32 {
// CHECK:             %[[IF_0:.*]] = scf.if %[[ARG0]] -> (f32) {
// CHECK:               %[[IF_1:.*]] = scf.if %[[ARG1]] -> (f32) {
// CHECK:                 scf.yield %[[ARG2]] : f32
// CHECK:               } else {
// CHECK:                 scf.yield %[[ARG3]] : f32
// CHECK:               }
// CHECK:               scf.yield %[[IF_1]] : f32
// CHECK:             } else {
// CHECK:               scf.yield %[[ARG3]] : f32
// CHECK:             }
// CHECK:             scf.yield %[[IF_0]] : f32
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : f32
// CHECK:         }
func.func @break_in_nested_if(%c1: i1, %c2: i1, %v1: f32, %v2: f32) -> f32 {
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok: token):
    scf.if %c1 {
      scf.if %c2 {
        scf.break %tok, %v1 : f32
      }
    }
    scf.yield %v2 : f32
  }
  return %0 : f32
}

// -----

// `scf.break` in the else-branch should be handled symmetrically to the
// then-branch case.

// CHECK-LABEL:   func.func @break_in_else(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_]*]]: f32) -> f32 {
// CHECK:           %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> f32 {
// CHECK:             %[[IF_0:.*]] = scf.if %[[ARG0]] -> (f32) {
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[ARG2]], %[[ARG2]] : f32
// CHECK:               scf.yield %[[ADDF_0]] : f32
// CHECK:             } else {
// CHECK:               scf.yield %[[ARG1]] : f32
// CHECK:             }
// CHECK:             scf.yield %[[IF_0]] : f32
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : f32
// CHECK:         }
func.func @break_in_else(%cond: i1, %v1: f32, %v2: f32) -> f32 {
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok: token):
    scf.if %cond {
    } else {
      scf.break %tok, %v1 : f32
    }
    %r = arith.addf %v2, %v2 : f32
    scf.yield %r : f32
  }
  return %0 : f32
}

// -----

// `scf.execute_region` ops with no result and no break inside are left
// untouched (other than the token block argument, which we keep around -- it
// can be eliminated by canonicalization).

// CHECK-LABEL:   func.func @nontoken_execute_region() {
// CHECK:           scf.execute_region {
// CHECK:             "test.foo"() : () -> ()
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @nontoken_execute_region() {
  scf.execute_region {
    "test.foo"() : () -> ()
    scf.yield
  }
  return
}

// -----

// Both branches of an `scf.if` break to the same target. Lowering converts
// each break into the corresponding "true / break-val" yield, and the if
// becomes effectively unconditional (the parent yield still forwards).

// CHECK-LABEL:   func.func @break_in_both_branches_same_target(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_]*]]: f32) -> f32 {
// CHECK:           %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> f32 {
// CHECK:             %[[IF_0:.*]] = scf.if %[[ARG0]] -> (f32) {
// CHECK:               scf.yield %[[ARG1]] : f32
// CHECK:             } else {
// CHECK:               scf.yield %[[ARG2]] : f32
// CHECK:             }
// CHECK:             scf.yield %[[IF_0]] : f32
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : f32
// CHECK:         }
func.func @break_in_both_branches_same_target(%cond: i1, %v1: f32, %v2: f32)
    -> f32 {
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok: token):
    scf.if %cond {
      scf.break %tok, %v1 : f32
    } else {
      scf.break %tok, %v2 : f32
    }
    // Unreachable trailing op (both branches break) — still legal IR.
    %unused = arith.constant 0.0 : f32
    scf.yield %unused : f32
  }
  return %0 : f32
}

// -----

// One branch breaks, the other yields. After lowering the if has both
// branches yielding the same f32 (one from the break value, one from the
// fall-through path).

// CHECK-LABEL:   func.func @break_then_yield_else(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_]*]]: f32) -> f32 {
// CHECK:           %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> f32 {
// CHECK:             %[[IF_0:.*]] = scf.if %[[ARG0]] -> (f32) {
// CHECK:               scf.yield %[[ARG1]] : f32
// CHECK:             } else {
// CHECK:               scf.yield %[[ARG2]] : f32
// CHECK:             }
// CHECK:             scf.yield %[[IF_0]] : f32
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : f32
// CHECK:         }
func.func @break_then_yield_else(%cond: i1, %v1: f32, %v2: f32) -> f32 {
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok: token):
    %y = scf.if %cond -> f32 {
      scf.break %tok, %v1 : f32
    } else {
      scf.yield %v2 : f32
    }
    scf.yield %y : f32
  }
  return %0 : f32
}

// -----

// The "then" branch breaks to the outer execute_region and the "else" branch
// breaks to the inner (self) one. The inner execute_region grows by an
// (i1, f32) pair for the outer target; the self-break populates the original
// f32 slot directly.

// A single guarded re-break to the outer target, immediately re-lowered by
// the outer pass into a yielding `scf.if`.
// CHECK-LABEL:   func.func @break_two_branches_two_targets(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_]*]]: f32) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant false
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant true
// CHECK:           %[[POISON_0:.*]] = ub.poison : f32
// CHECK:           %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> f32 {
// CHECK:             %[[EXECUTE_REGION_1:.*]]:3 = scf.execute_region -> (f32, i1, f32) {
// CHECK:               %[[IF_0:.*]]:3 = scf.if %[[ARG0]] -> (f32, i1, f32) {
// CHECK:                 scf.yield %[[POISON_0]], %[[CONSTANT_1]], %[[ARG1]] : f32, i1, f32
// CHECK:               } else {
// CHECK:                 scf.yield %[[ARG2]], %[[CONSTANT_0]], %[[POISON_0]] : f32, i1, f32
// CHECK:               }
// CHECK:               scf.yield %[[VAL_0:.*]]#0, %[[VAL_0]]#1, %[[VAL_0]]#2 : f32, i1, f32
// CHECK:             }
// CHECK:             %[[IF_1:.*]] = scf.if %[[VAL_1:.*]]#1 -> (f32) {
// CHECK:               scf.yield %[[VAL_1]]#2 : f32
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_2:.*]]#0 : f32
// CHECK:             }
// CHECK:             scf.yield %[[IF_1]] : f32
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : f32
// CHECK:         }
func.func @break_two_branches_two_targets(%c: i1, %v1: f32, %v2: f32) -> f32 {
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok_outer: token):
    %r = scf.execute_region -> f32 {
    ^bb0(%tok_inner: token):
      %x = scf.if %c -> f32 {
        scf.break %tok_outer, %v1 : f32
      } else {
        scf.break %tok_inner, %v2 : f32
      }
      scf.yield %x : f32
    }
    scf.yield %r : f32
  }
  return %0 : f32
}

// -----

// Multiple distinct outer break targets: an inner execute_region contains two
// `scf.if` ops, each of which breaks to a different ancestor execute_region.
// The inner execute_region's result list grows by (1 + |target.results|)
// slots per outer target -- here, 1 + 1 (f32) + 1 + 1 (i32) = 4 extra results.
// All breaks are eliminated; each outer target is handled by the post-region
// guarded `scf.if { scf.break }` it generates (which is then re-lowered by
// the outer pass).

// The lowering augments the middle execute_region (original i32) with one
// `(i1, f32)` pair for the outermost target, and the innermost
// execute_region (original f32) with both an `(i1, i32)` pair for the
// middle target and an `(i1, f32)` pair for the outermost target. The
// exact order in which slot pairs are appended is an implementation
// detail of the lowering pass; what matters here is that both targets
// are threaded out cleanly and turn into guarded re-breaks that
// themselves get resolved into forwarding `scf.if`s.

//
// Middle execute_region: original i32 result + (i1, f32) for the
// outermost target.
//
// Innermost execute_region: original f32 result + (i1, i32) for the
// middle target + (i1, f32) for the outermost target.
//
// The chain `scf.if`s inside the innermost body all carry the same
// (f32, i1, i32, i1, f32) shape. Both original breaks have become
// augmented yields (with poison for the slots the branch doesn't drive).
//
// First post-region guard (in the middle ER's body): re-break for the
// outermost target, already turned into a forwarding `scf.if -> i32` by
// the resolve step.
//
// Second post-region guard (in the outermost ER's body): re-break for
// the outermost target, resolved into a forwarding `scf.if -> f32`.
// CHECK-LABEL:   func.func @break_to_two_outer_targets(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG3:[a-zA-Z0-9_]*]]: i32,
// CHECK-SAME:      %[[ARG4:[a-zA-Z0-9_]*]]: f32) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant false
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant true
// CHECK:           %[[POISON_I32:.*]] = ub.poison : i32
// CHECK:           %[[POISON_I1:.*]] = ub.poison : i1
// CHECK:           %[[POISON_F32:.*]] = ub.poison : f32
// CHECK:           %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> f32 {
// CHECK:             %[[EXECUTE_REGION_1:.*]]:3 = scf.execute_region -> (i32, i1, f32) {
// CHECK:               %[[EXECUTE_REGION_2:.*]]:5 = scf.execute_region -> (f32, i1, i32, i1, f32) {
// CHECK:                 %[[IF_0:.*]]:5 = scf.if %[[ARG0]] -> (f32, i1, i32, i1, f32) {
// CHECK:                   scf.yield %[[POISON_F32]], %[[POISON_I1]], %[[POISON_I32]], %[[CONSTANT_1]], %[[ARG2]] : f32, i1, i32, i1, f32
// CHECK:                 } else {
// CHECK:                   %[[IF_1:.*]]:3 = scf.if %[[ARG1]] -> (f32, i1, i32) {
// CHECK:                     scf.yield %[[POISON_F32]], %[[CONSTANT_1]], %[[ARG3]] : f32, i1, i32
// CHECK:                   } else {
// CHECK:                     scf.yield %[[ARG4]], %[[CONSTANT_0]], %[[POISON_I32]] : f32, i1, i32
// CHECK:                   }
// CHECK:                   scf.yield %[[IF_1]]#0, %[[IF_1]]#1, %[[IF_1]]#2, %[[CONSTANT_0]], %[[POISON_F32]] : f32, i1, i32, i1, f32
// CHECK:                 }
// CHECK:                 scf.yield %[[IF_0]]#0, %[[IF_0]]#1, %[[IF_0]]#2, %[[IF_0]]#3, %[[IF_0]]#4 : f32, i1, i32, i1, f32
// CHECK:               }
// CHECK:               %[[IF_2:.*]]:3 = scf.if %[[EXECUTE_REGION_2]]#3 -> (i32, i1, f32) {
// CHECK:                 scf.yield %[[POISON_I32]], %[[CONSTANT_1]], %[[EXECUTE_REGION_2]]#4 : i32, i1, f32
// CHECK:               } else {
// CHECK:                 %[[IF_3:.*]] = scf.if %[[EXECUTE_REGION_2]]#1 -> (i32) {
// CHECK:                   scf.yield %[[EXECUTE_REGION_2]]#2 : i32
// CHECK:                 } else {
// CHECK:                   "test.use_f32"(%[[EXECUTE_REGION_2]]#0) : (f32) -> ()
// CHECK:                   scf.yield %[[ARG3]] : i32
// CHECK:                 }
// CHECK:                 scf.yield %[[IF_3]], %[[CONSTANT_0]], %[[POISON_F32]] : i32, i1, f32
// CHECK:               }
// CHECK:               scf.yield %[[IF_2]]#0, %[[IF_2]]#1, %[[IF_2]]#2 : i32, i1, f32
// CHECK:             }
// CHECK:             %[[IF_4:.*]] = scf.if %[[EXECUTE_REGION_1]]#1 -> (f32) {
// CHECK:               scf.yield %[[EXECUTE_REGION_1]]#2 : f32
// CHECK:             } else {
// CHECK:               "test.use_i32"(%[[EXECUTE_REGION_1]]#0) : (i32) -> ()
// CHECK:               scf.yield %[[ARG2]] : f32
// CHECK:             }
// CHECK:             scf.yield %[[IF_4]] : f32
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : f32
// CHECK:         }
func.func @break_to_two_outer_targets(%c1: i1, %c2: i1, %v1: f32, %v2: i32,
                                       %v3: f32) -> f32 {
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok1: token):
    %r = scf.execute_region -> i32 {
    ^bb0(%tok2: token):
      %s = scf.execute_region -> f32 {
      ^bb0(%tok3: token):
        scf.if %c1 {
          scf.break %tok1, %v1 : f32
        }
        scf.if %c2 {
          scf.break %tok2, %v2 : i32
        }
        scf.yield %v3 : f32
      }
      "test.use_f32"(%s) : (f32) -> ()
      scf.yield %v2 : i32
    }
    "test.use_i32"(%r) : (i32) -> ()
    scf.yield %v1 : f32
  }
  return %0 : f32
}

// -----

// A self-break inside a then-branch and another self-break inside the else-
// branch of the SAME if both target the inner execute_region. No outer
// targets are involved, so no augmentation suffix is added; both breaks
// become plain `scf.yield`s.

// CHECK-LABEL:   func.func @both_branches_self_break(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_]*]]: f32) -> f32 {
// CHECK:           %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> f32 {
// CHECK:             %[[IF_0:.*]] = scf.if %[[ARG0]] -> (f32) {
// CHECK:               scf.yield %[[ARG1]] : f32
// CHECK:             } else {
// CHECK:               scf.yield %[[ARG2]] : f32
// CHECK:             }
// CHECK:             scf.yield %[[IF_0]] : f32
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : f32
// CHECK:         }
func.func @both_branches_self_break(%c: i1, %v1: f32, %v2: f32) -> f32 {
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok: token):
    scf.if %c {
      scf.break %tok, %v1 : f32
    } else {
      scf.break %tok, %v2 : f32
    }
    scf.yield %v1 : f32
  }
  return %0 : f32
}

// -----

// A `scf.break` carrying multiple result values.

// CHECK-LABEL:   func.func @break_multi_result(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_]*]]: i32,
// CHECK-SAME:      %[[ARG3:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG4:[a-zA-Z0-9_]*]]: i32) -> (f32, i32) {
// CHECK:           %[[EXECUTE_REGION_0:.*]]:2 = scf.execute_region -> (f32, i32) {
// CHECK:             %[[IF_0:.*]]:2 = scf.if %[[ARG0]] -> (f32, i32) {
// CHECK:               scf.yield %[[ARG1]], %[[ARG2]] : f32, i32
// CHECK:             } else {
// CHECK:               scf.yield %[[ARG3]], %[[ARG4]] : f32, i32
// CHECK:             }
// CHECK:             scf.yield %[[VAL_0:.*]]#0, %[[VAL_0]]#1 : f32, i32
// CHECK:           }
// CHECK:           return %[[VAL_1:.*]]#0, %[[VAL_1]]#1 : f32, i32
// CHECK:         }
func.func @break_multi_result(%c: i1, %a: f32, %b: i32, %x: f32, %y: i32)
    -> (f32, i32) {
  %0:2 = scf.execute_region -> (f32, i32) {
  ^bb0(%tok: token):
    scf.if %c {
      scf.break %tok, %a, %b : f32, i32
    }
    scf.yield %x, %y : f32, i32
  }
  return %0#0, %0#1 : f32, i32
}

// -----

// A zero-result early exit (a `scf.break` with no values).

// CHECK-LABEL:   func.func @break_zero_result(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: i1) {
// CHECK:           scf.execute_region {
// CHECK:             scf.if %[[ARG0]] {
// CHECK:             } else {
// CHECK:               "test.body"() : () -> ()
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @break_zero_result(%c: i1) {
  scf.execute_region {
  ^bb0(%tok: token):
    scf.if %c {
      scf.break %tok
    }
    "test.body"() : () -> ()
    scf.yield
  }
  return
}

// -----

// A break nested in two `scf.if`s that targets the *outer* execute_region
// (combines flag-based hoisting with if-sinking/tail-duplication).

// CHECK-LABEL:   func.func @break_outer_through_nested_if(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG3:[a-zA-Z0-9_]*]]: f32) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant false
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant true
// CHECK:           %[[POISON_0:.*]] = ub.poison : f32
// CHECK:           %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> f32 {
// CHECK:             %[[EXECUTE_REGION_1:.*]]:3 = scf.execute_region -> (f32, i1, f32) {
// CHECK:               %[[IF_0:.*]]:3 = scf.if %[[ARG0]] -> (f32, i1, f32) {
// CHECK:                 %[[IF_1:.*]]:3 = scf.if %[[ARG1]] -> (f32, i1, f32) {
// CHECK:                   scf.yield %[[POISON_0]], %[[CONSTANT_1]], %[[ARG2]] : f32, i1, f32
// CHECK:                 } else {
// CHECK:                   scf.yield %[[ARG3]], %[[CONSTANT_0]], %[[POISON_0]] : f32, i1, f32
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_0:.*]]#0, %[[VAL_0]]#1, %[[VAL_0]]#2 : f32, i1, f32
// CHECK:               } else {
// CHECK:                 scf.yield %[[ARG3]], %[[CONSTANT_0]], %[[POISON_0]] : f32, i1, f32
// CHECK:               }
// CHECK:               scf.yield %[[VAL_1:.*]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2 : f32, i1, f32
// CHECK:             }
// CHECK:             %[[IF_2:.*]] = scf.if %[[VAL_2:.*]]#1 -> (f32) {
// CHECK:               scf.yield %[[VAL_2]]#2 : f32
// CHECK:             } else {
// CHECK:               %[[VAL_3:.*]] = "test.use"(%[[VAL_4:.*]]#0) : (f32) -> f32
// CHECK:               scf.yield %[[VAL_3]] : f32
// CHECK:             }
// CHECK:             scf.yield %[[IF_2]] : f32
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : f32
// CHECK:         }
func.func @break_outer_through_nested_if(%c1: i1, %c2: i1, %v1: f32, %v2: f32)
    -> f32 {
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok: token):
    %r = scf.execute_region -> f32 {
    ^bb1(%tok1: token):
      scf.if %c1 {
        scf.if %c2 {
          scf.break %tok, %v1 : f32
        }
      }
      scf.yield %v2 : f32
    }
    %u = "test.use"(%r) : (f32) -> f32
    scf.yield %u : f32
  }
  return %0 : f32
}

// -----

// The continuation after a breaking `scf.if` has side effects and consumes the
// if's result; sinking must remap the result to the fall-through value.

// CHECK-LABEL:   func.func @break_sink_with_use(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_]*]]: f32) -> f32 {
// CHECK:           %[[EXECUTE_REGION_0:.*]] = scf.execute_region -> f32 {
// CHECK:             %[[IF_0:.*]] = scf.if %[[ARG0]] -> (f32) {
// CHECK:               scf.yield %[[ARG1]] : f32
// CHECK:             } else {
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[ARG2]], %[[ARG2]] : f32
// CHECK:               "test.sink"(%[[ADDF_0]]) : (f32) -> ()
// CHECK:               scf.yield %[[ADDF_0]] : f32
// CHECK:             }
// CHECK:             scf.yield %[[IF_0]] : f32
// CHECK:           }
// CHECK:           return %[[EXECUTE_REGION_0]] : f32
// CHECK:         }
func.func @break_sink_with_use(%c: i1, %v1: f32, %v2: f32) -> f32 {
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok: token):
    %y = scf.if %c -> f32 {
      scf.break %tok, %v1 : f32
    } else {
      scf.yield %v2 : f32
    }
    %z = arith.addf %y, %y : f32
    "test.sink"(%z) : (f32) -> ()
    scf.yield %z : f32
  }
  return %0 : f32
}

// -----

// When an execute_region is widened, every break targeting its token must be
// extended, including breaks nested under another token-bearing execute_region.
// Otherwise the nested break would still have the old operand list for the
// widened target.

// CHECK-LABEL:   func.func @extend_nested_break_to_widened_region(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_]*]]: i1,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG3:[a-zA-Z0-9_]*]]: f32,
// CHECK-SAME:      %[[ARG4:[a-zA-Z0-9_]*]]: f32) -> f32 {
// CHECK:           %[[FALSE:.*]] = arith.constant false
// CHECK:           %[[TRUE:.*]] = arith.constant true
// CHECK:           %[[POISON:.*]] = ub.poison : f32
// CHECK:           %[[OUTER:.*]] = scf.execute_region -> f32 {
// CHECK:             %[[INNER:.*]]:3 = scf.execute_region -> (f32, i1, f32) {
// CHECK:               %[[IF_OUTER:.*]]:3 = scf.if %[[ARG0]] -> (f32, i1, f32) {
// CHECK:                 scf.yield %[[POISON]], %[[TRUE]], %[[ARG2]] : f32, i1, f32
// CHECK:               } else {
// CHECK:                 %[[DEEP:.*]]:3 = scf.execute_region -> (f32, i1, f32) {
// CHECK:                   %[[IF_DEEP:.*]]:3 = scf.if %[[ARG1]] -> (f32, i1, f32) {
// CHECK:                     scf.yield %[[POISON]], %[[TRUE]], %[[ARG3]] : f32, i1, f32
// CHECK:                   } else {
// CHECK:                     scf.yield %[[ARG4]], %[[FALSE]], %[[POISON]] : f32, i1, f32
// CHECK:                   }
// CHECK:                   scf.yield %[[IF_DEEP]]#0, %[[IF_DEEP]]#1, %[[IF_DEEP]]#2 : f32, i1, f32
// CHECK:                 }
// CHECK:                 %[[IF_INNER_REBREAK:.*]] = scf.if %[[DEEP]]#1 -> (f32) {
// CHECK:                   scf.yield %[[DEEP]]#2 : f32
// CHECK:                 } else {
// CHECK:                   scf.yield %[[DEEP]]#0 : f32
// CHECK:                 }
// CHECK:                 scf.yield %[[IF_INNER_REBREAK]], %[[FALSE]], %[[POISON]] : f32, i1, f32
// CHECK:               }
// CHECK:               scf.yield %[[IF_OUTER]]#0, %[[IF_OUTER]]#1, %[[IF_OUTER]]#2 : f32, i1, f32
// CHECK:             }
// CHECK:             %[[IF_OUTER_REBREAK:.*]] = scf.if %[[INNER]]#1 -> (f32) {
// CHECK:               scf.yield %[[INNER]]#2 : f32
// CHECK:             } else {
// CHECK:               scf.yield %[[INNER]]#0 : f32
// CHECK:             }
// CHECK:             scf.yield %[[IF_OUTER_REBREAK]] : f32
// CHECK:           }
// CHECK:           return %[[OUTER]] : f32
// CHECK:         }
func.func @extend_nested_break_to_widened_region(%c1: i1, %c2: i1, %v1: f32,
                                                 %v2: f32, %v3: f32) -> f32 {
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok_outer: token):
    %inner = scf.execute_region -> f32 {
    ^bb0(%tok_inner: token):
      scf.if %c1 {
        scf.break %tok_outer, %v1 : f32
      }
      %deep = scf.execute_region -> f32 {
      ^bb0(%tok_deep: token):
        scf.if %c2 {
          scf.break %tok_inner, %v2 : f32
        }
        scf.yield %v3 : f32
      }
      scf.yield %deep : f32
    }
    scf.yield %inner : f32
  }
  return %0 : f32
}

