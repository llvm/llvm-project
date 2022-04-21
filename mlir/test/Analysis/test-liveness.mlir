// RUN: mlir-opt %s -pass-pipeline="func.func(test-print-liveness)" -split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : func_empty
func.func @func_empty() {
  // CHECK: Block: 0
  // CHECK-NEXT: LiveIn:{{ *$}}
  // CHECK-NEXT: LiveOut:{{ *$}}
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: EndLiveness
  return
}

// -----

// CHECK-LABEL: Testing : func_simpleBranch
func.func @func_simpleBranch(%arg0: i32, %arg1 : i32) -> i32 {
  // CHECK: Block: 0
  // CHECK-NEXT: LiveIn:{{ *$}}
  // CHECK-NEXT: LiveOut: arg0@0 arg1@0
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: EndLiveness
  cf.br ^exit
^exit:
  // CHECK: Block: 1
  // CHECK-NEXT: LiveIn: arg0@0 arg1@0
  // CHECK-NEXT: LiveOut:{{ *$}}
  // CHECK-NEXT: BeginLiveness
  // CHECK: val_2
  // CHECK-NEXT:     %0 = arith.addi
  // CHECK-NEXT:     return
  // CHECK-NEXT: EndLiveness
  %result = arith.addi %arg0, %arg1 : i32
  return %result : i32
}

// -----

// CHECK-LABEL: Testing : func_condBranch
func.func @func_condBranch(%cond : i1, %arg1: i32, %arg2 : i32) -> i32 {
  // CHECK: Block: 0
  // CHECK-NEXT: LiveIn:{{ *$}}
  // CHECK-NEXT: LiveOut: arg1@0 arg2@0
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: EndLiveness
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  // CHECK: Block: 1
  // CHECK-NEXT: LiveIn: arg1@0 arg2@0
  // CHECK-NEXT: LiveOut: arg1@0 arg2@0
  cf.br ^exit
^bb2:
  // CHECK: Block: 2
  // CHECK-NEXT: LiveIn: arg1@0 arg2@0
  // CHECK-NEXT: LiveOut: arg1@0 arg2@0
  cf.br ^exit
^exit:
  // CHECK: Block: 3
  // CHECK-NEXT: LiveIn: arg1@0 arg2@0
  // CHECK-NEXT: LiveOut:{{ *$}}
  // CHECK-NEXT: BeginLiveness
  // CHECK: val_3
  // CHECK-NEXT:     %0 = arith.addi
  // CHECK-NEXT:     return
  // CHECK-NEXT: EndLiveness
  %result = arith.addi %arg1, %arg2 : i32
  return %result : i32
}

// -----

// CHECK-LABEL: Testing : func_loop
func.func @func_loop(%arg0 : i32, %arg1 : i32) -> i32 {
  // CHECK: Block: 0
  // CHECK-NEXT: LiveIn:{{ *$}}
  // CHECK-NEXT: LiveOut: arg1@0
  %const0 = arith.constant 0 : i32
  cf.br ^loopHeader(%const0, %arg0 : i32, i32)
^loopHeader(%counter : i32, %i : i32):
  // CHECK: Block: 1
  // CHECK-NEXT: LiveIn: arg1@0
  // CHECK-NEXT: LiveOut: arg1@0 arg0@1
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: val_5
  // CHECK-NEXT:     %2 = arith.cmpi
  // CHECK-NEXT:     cf.cond_br
  // CHECK-NEXT: EndLiveness
  %lessThan = arith.cmpi slt, %counter, %arg1 : i32
  cf.cond_br %lessThan, ^loopBody(%i : i32), ^exit(%i : i32)
^loopBody(%val : i32):
  // CHECK: Block: 2
  // CHECK-NEXT: LiveIn: arg1@0 arg0@1
  // CHECK-NEXT: LiveOut: arg1@0
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: val_7
  // CHECK-NEXT:     %c
  // CHECK-NEXT:     %4 = arith.addi
  // CHECK-NEXT:     %5 = arith.addi
  // CHECK-NEXT: val_8
  // CHECK-NEXT:     %4 = arith.addi
  // CHECK-NEXT:     %5 = arith.addi
  // CHECK-NEXT:     cf.br
  // CHECK: EndLiveness
  %const1 = arith.constant 1 : i32
  %inc = arith.addi %val, %const1 : i32
  %inc2 = arith.addi %counter, %const1 : i32
  cf.br ^loopHeader(%inc, %inc2 : i32, i32)
^exit(%sum : i32):
  // CHECK: Block: 3
  // CHECK-NEXT: LiveIn: arg1@0
  // CHECK-NEXT: LiveOut:{{ *$}}
  %result = arith.addi %sum, %arg1 : i32
  return %result : i32
}

// -----

// CHECK-LABEL: Testing : func_ranges
func.func @func_ranges(%cond : i1, %arg1 : i32, %arg2 : i32, %arg3 : i32) -> i32 {
  // CHECK: Block: 0
  // CHECK-NEXT: LiveIn:{{ *$}}
  // CHECK-NEXT: LiveOut: arg2@0 val_9 val_10
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: val_4
  // CHECK-NEXT:    %0 = arith.addi
  // CHECK-NEXT:    %c
  // CHECK-NEXT:    %1 = arith.addi
  // CHECK-NEXT:    %2 = arith.addi
  // CHECK-NEXT:    %3 = arith.muli
  // CHECK-NEXT: val_5
  // CHECK-NEXT:    %c
  // CHECK-NEXT:    %1 = arith.addi
  // CHECK-NEXT:    %2 = arith.addi
  // CHECK-NEXT:    %3 = arith.muli
  // CHECK-NEXT:    %4 = arith.muli
  // CHECK-NEXT:    %5 = arith.addi
  // CHECK-NEXT: val_6
  // CHECK-NEXT:    %1 = arith.addi
  // CHECK-NEXT:    %2 = arith.addi
  // CHECK-NEXT:    %3 = arith.muli
  // CHECK-NEXT: val_7
  // CHECK-NEXT    %2 = arith.addi
  // CHECK-NEXT    %3 = arith.muli
  // CHECK-NEXT    %4 = arith.muli
  // CHECK:      val_8
  // CHECK-NEXT:    %3 = arith.muli
  // CHECK-NEXT:    %4 = arith.muli
  // CHECK-NEXT: val_9
  // CHECK-NEXT:    %4 = arith.muli
  // CHECK-NEXT:    %5 = arith.addi
  // CHECK-NEXT:    cf.cond_br
  // CHECK-NEXT:    %c
  // CHECK-NEXT:    %6 = arith.muli
  // CHECK-NEXT:    %7 = arith.muli
  // CHECK-NEXT:    %8 = arith.addi
  // CHECK-NEXT: val_10
  // CHECK-NEXT:    %5 = arith.addi
  // CHECK-NEXT:    cf.cond_br
  // CHECK-NEXT:    %7
  // CHECK: EndLiveness
  %0 = arith.addi %arg1, %arg2 : i32
  %const1 = arith.constant 1 : i32
  %1 = arith.addi %const1, %arg2 : i32
  %2 = arith.addi %const1, %arg3 : i32
  %3 = arith.muli %0, %1 : i32
  %4 = arith.muli %3, %2 : i32
  %5 = arith.addi %4, %const1 : i32
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  // CHECK: Block: 1
  // CHECK-NEXT: LiveIn: arg2@0 val_9
  // CHECK-NEXT: LiveOut: arg2@0
  %const4 = arith.constant 4 : i32
  %6 = arith.muli %4, %const4 : i32
  cf.br ^exit(%6 : i32)

^bb2:
  // CHECK: Block: 2
  // CHECK-NEXT: LiveIn: arg2@0 val_9 val_10
  // CHECK-NEXT: LiveOut: arg2@0
  %7 = arith.muli %4, %5 : i32
  %8 = arith.addi %4, %arg2 : i32
  cf.br ^exit(%8 : i32)

^exit(%sum : i32):
  // CHECK: Block: 3
  // CHECK-NEXT: LiveIn: arg2@0
  // CHECK-NEXT: LiveOut:{{ *$}}
  %result = arith.addi %sum, %arg2 : i32
  return %result : i32
}

// -----

// CHECK-LABEL: Testing : nested_region

func.func @nested_region(
  %arg0 : index, %arg1 : index, %arg2 : index,
  %arg3 : i32, %arg4 : i32, %arg5 : i32,
  %buffer : memref<i32>) -> i32 {
  // CHECK: Block: 0
  // CHECK-NEXT: LiveIn:{{ *$}}
  // CHECK-NEXT: LiveOut:{{ *$}}
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: val_7
  // CHECK-NEXT:    %0 = arith.addi
  // CHECK-NEXT:    %1 = arith.addi
  // CHECK-NEXT:    scf.for
  // CHECK:         // %2 = arith.addi
  // CHECK-NEXT:    %3 = arith.addi
  // CHECK-NEXT: val_8
  // CHECK-NEXT:    %1 = arith.addi
  // CHECK-NEXT:    scf.for
  // CHECK:         // return %1
  // CHECK: EndLiveness
  %0 = arith.addi %arg3, %arg4 : i32
  %1 = arith.addi %arg4, %arg5 : i32
  scf.for %arg6 = %arg0 to %arg1 step %arg2 {
    // CHECK: Block: 1
    // CHECK-NEXT: LiveIn: arg5@0 arg6@0 val_7
    // CHECK-NEXT: LiveOut:{{ *$}}
    %2 = arith.addi %0, %arg5 : i32
    %3 = arith.addi %2, %0 : i32
    memref.store %3, %buffer[] : memref<i32>
  }
  return %1 : i32
}

// -----

// CHECK-LABEL: Testing : nested_region2

func.func @nested_region2(
  // CHECK: Block: 0
  // CHECK-NEXT: LiveIn:{{ *$}}
  // CHECK-NEXT: LiveOut:{{ *$}}
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: val_7
  // CHECK-NEXT:    %0 = arith.addi
  // CHECK-NEXT:    %1 = arith.addi
  // CHECK-NEXT:    scf.for
  // CHECK:         // %2 = arith.addi
  // CHECK-NEXT:    scf.for
  // CHECK:         // %3 = arith.addi
  // CHECK-NEXT: val_8
  // CHECK-NEXT:    %1 = arith.addi
  // CHECK-NEXT:    scf.for
  // CHECK:         // return %1
  // CHECK: EndLiveness
  %arg0 : index, %arg1 : index, %arg2 : index,
  %arg3 : i32, %arg4 : i32, %arg5 : i32,
  %buffer : memref<i32>) -> i32 {
  %0 = arith.addi %arg3, %arg4 : i32
  %1 = arith.addi %arg4, %arg5 : i32
  scf.for %arg6 = %arg0 to %arg1 step %arg2 {
    // CHECK: Block: 1
    // CHECK-NEXT: LiveIn: arg0@0 arg1@0 arg2@0 arg5@0 arg6@0 val_7
    // CHECK-NEXT: LiveOut:{{ *$}}
    // CHECK-NEXT: BeginLiveness
    // CHECK-NEXT: val_10
    // CHECK-NEXT:    %2 = arith.addi
    // CHECK-NEXT:    scf.for
    // CHECK:         // %3 = arith.addi
    // CHECK: EndLiveness
    %2 = arith.addi %0, %arg5 : i32
    scf.for %arg7 = %arg0 to %arg1 step %arg2 {
      %3 = arith.addi %2, %0 : i32
      memref.store %3, %buffer[] : memref<i32>
    }
  }
  return %1 : i32
}

// -----

// CHECK-LABEL: Testing : nested_region3

func.func @nested_region3(
  // CHECK: Block: 0
  // CHECK-NEXT: LiveIn:{{ *$}}
  // CHECK-NEXT: LiveOut: arg0@0 arg1@0 arg2@0 arg6@0 val_7 val_8
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: val_7
  // CHECK-NEXT:    %0 = arith.addi
  // CHECK-NEXT:    %1 = arith.addi
  // CHECK-NEXT:    scf.for
  // CHECK:         // cf.br ^bb1
  // CHECK-NEXT:    %2 = arith.addi
  // CHECK-NEXT:    scf.for
  // CHECK:         // %2 = arith.addi
  // CHECK: EndLiveness
  %arg0 : index, %arg1 : index, %arg2 : index,
  %arg3 : i32, %arg4 : i32, %arg5 : i32,
  %buffer : memref<i32>) -> i32 {
  %0 = arith.addi %arg3, %arg4 : i32
  %1 = arith.addi %arg4, %arg5 : i32
  scf.for %arg6 = %arg0 to %arg1 step %arg2 {
    // CHECK: Block: 1
    // CHECK-NEXT: LiveIn: arg5@0 arg6@0 val_7
    // CHECK-NEXT: LiveOut:{{ *$}}
    %2 = arith.addi %0, %arg5 : i32
    memref.store %2, %buffer[] : memref<i32>
  }
  cf.br ^exit

^exit:
  // CHECK: Block: 2
  // CHECK-NEXT: LiveIn: arg0@0 arg1@0 arg2@0 arg6@0 val_7 val_8
  // CHECK-NEXT: LiveOut:{{ *$}}
  scf.for %arg7 = %arg0 to %arg1 step %arg2 {
    // CHECK: Block: 3
    // CHECK-NEXT: LiveIn: arg6@0 val_7 val_8
    // CHECK-NEXT: LiveOut:{{ *$}}
    %2 = arith.addi %0, %1 : i32
    memref.store %2, %buffer[] : memref<i32>
  }
  return %1 : i32
}
