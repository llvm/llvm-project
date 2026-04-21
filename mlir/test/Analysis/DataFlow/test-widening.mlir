// Exercises the merge-site widening hook across the four sparse/dense x
// forward/backward analysis variants. Uses an infinite-height counter
// lattice with a small widening budget; loops trip widening, straight-line
// code stays below it.
//
// RUN: mlir-opt %s -split-input-file \
// RUN:   --test-widening-analysis=variant=sparse-forward  2>&1 | FileCheck %s --check-prefix=SF
// RUN: mlir-opt %s -split-input-file \
// RUN:   --test-widening-analysis=variant=sparse-backward 2>&1 | FileCheck %s --check-prefix=SB
// RUN: mlir-opt %s -split-input-file \
// RUN:   --test-widening-analysis=variant=dense-forward   2>&1 | FileCheck %s --check-prefix=DF
// RUN: mlir-opt %s -split-input-file \
// RUN:   --test-widening-analysis=variant=dense-backward  2>&1 | FileCheck %s --check-prefix=DB

//===----------------------------------------------------------------------===//
// Straight-line code: counter stays bounded, no widening.
//===----------------------------------------------------------------------===//

// SF: tag=straight: results=[count={{[0-9]+}}]
// SB: tag=straight: operands=[count={{[0-9]+}}, count={{[0-9]+}}]
// DF: tag=straight: count={{[0-9]+}}
// DB: tag=straight: count={{[0-9]+}}

func.func @straight(%x: i32) -> i32 {
  %a = arith.addi %x, %x : i32
  %b = arith.addi %a, %a : i32
  %c = arith.addi %b, %b : i32
  %d = arith.addi %c, %c {tag = "straight"} : i32
  return %d : i32
}

// -----

//===----------------------------------------------------------------------===//
// Structured loop (scf.while with dynamic bound): widening fires on the
// loop-carried value and propagates through the body.
//===----------------------------------------------------------------------===//

// SF: tag=scf_body: results=[widened]
// SB: tag=scf_body: operands=[widened, {{widened|count=[0-9]+}}]
// DF: tag=scf_body: widened
// DB: tag=scf_body: widened

func.func @scf_while(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %init = arith.cmpi slt, %c0, %n : i32
  %r:2 = scf.while (%acc = %c0, %cond = %init) : (i32, i1) -> (i32, i1) {
    scf.condition(%cond) %acc, %cond : i32, i1
  } do {
  ^bb(%a: i32, %c: i1):
    %next = arith.addi %a, %c1 {tag = "scf_body"} : i32
    %nc = arith.cmpi slt, %next, %n : i32
    scf.yield %next, %nc : i32, i1
  }
  return %r#0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// Unstructured CFG cycle (cf.br / cf.cond_br): the merge-site hook fires at
// the block-arg join on the loop head without any loop detection. This is
// the architectural payoff over back-edge-only widening.
//===----------------------------------------------------------------------===//

// SF: tag=cfg_body: results=[widened]
// SB: tag=cfg_body: operands=[widened, {{widened|count=[0-9]+}}]
// DF: tag=cfg_body: widened
// DB: tag=cfg_body: widened

func.func @unstructured_cfg(%cond: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  cf.br ^head(%c0 : i32)
^head(%a: i32):
  %next = arith.addi %a, %c1 {tag = "cfg_body"} : i32
  cf.cond_br %cond, ^head(%next : i32), ^exit
^exit:
  return
}
