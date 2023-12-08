// RUN: mlir-opt %s -convert-cf-to-llvm | FileCheck %s

func.func @name(%flag: i32, %pred: i1){
    // Test cf.br lowering failure with type mismatch
    // CHECK: cf.br
    %c0 = arith.constant 0 : index
    cf.br ^bb1(%c0 : index)

  // Test cf.cond_br lowering failure with type mismatch in false_dest
  // CHECK: cf.cond_br
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
    %c1 = arith.constant 1 : i1
    %c2 = arith.constant 1 : index
    cf.cond_br %pred, ^bb2(%c1: i1), ^bb3(%c2: index)

  // Test cf.cond_br lowering failure with type mismatch in true_dest
  // CHECK: cf.cond_br
  ^bb2(%1: i1):
    %c3 = arith.constant 1 : i1
    %c4 = arith.constant 1 : index
    cf.cond_br %pred, ^bb3(%c4: index), ^bb2(%c3: i1)

  // Test cf.switch lowering failure with type mismatch in default case
  // CHECK: cf.switch
  ^bb3(%2: index):  // pred: ^bb1
    %c5 = arith.constant 1 : i1
    %c6 = arith.constant 1 : index
    cf.switch %flag : i32, [
      default: ^bb1(%c6 : index),
      42: ^bb4(%c5 : i1)
    ]

  // Test cf.switch lowering failure with type mismatch in non-default case
  // CHECK: cf.switch
  ^bb4(%3: i1):  // pred: ^bb1
    %c7 = arith.constant 1 : i1
    %c8 = arith.constant 1 : index
    cf.switch %flag : i32, [
      default: ^bb2(%c7 : i1),
      41: ^bb1(%c8 : index)
    ]
  }
