// RUN: mlir-opt %s -test-print-liveness -split-input-file 2>&1 | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: Testing : func_empty
func @func_empty() {
  // CHECK: Block: 0
  // CHECK-NEXT: LiveIn:{{ *$}}
  // CHECK-NEXT: LiveOut:{{ *$}}
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: EndLiveness
  return
}

// -----

// CHECK-LABEL: Testing : func_simpleBranch
func @func_simpleBranch(%arg0: i32, %arg1 : i32) -> i32 {
  // CHECK: Block: 0
  // CHECK-NEXT: LiveIn:{{ *$}}
  // CHECK-NEXT: LiveOut: arg0@0 arg1@0
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: EndLiveness
  br ^exit
^exit:
  // CHECK: Block: 1
  // CHECK-NEXT: LiveIn: arg0@0 arg1@0
  // CHECK-NEXT: LiveOut:{{ *$}}
  // CHECK-NEXT: BeginLiveness
  // CHECK: val_std.addi
  // CHECK-NEXT:     %0 = addi
  // CHECK-NEXT:     return
  // CHECK-NEXT: EndLiveness
  %result = addi %arg0, %arg1 : i32
  return %result : i32
}

// -----

// CHECK-LABEL: Testing : func_condBranch
func @func_condBranch(%cond : i1, %arg1: i32, %arg2 : i32) -> i32 {
  // CHECK: Block: 0
  // CHECK-NEXT: LiveIn:{{ *$}}
  // CHECK-NEXT: LiveOut: arg1@0 arg2@0
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: EndLiveness
  cond_br %cond, ^bb1, ^bb2
^bb1:
  // CHECK: Block: 1
  // CHECK-NEXT: LiveIn: arg1@0 arg2@0
  // CHECK-NEXT: LiveOut: arg1@0 arg2@0
  br ^exit
^bb2:
  // CHECK: Block: 2
  // CHECK-NEXT: LiveIn: arg1@0 arg2@0
  // CHECK-NEXT: LiveOut: arg1@0 arg2@0
  br ^exit
^exit:
  // CHECK: Block: 3
  // CHECK-NEXT: LiveIn: arg1@0 arg2@0
  // CHECK-NEXT: LiveOut:{{ *$}}
  // CHECK-NEXT: BeginLiveness
  // CHECK: val_std.addi
  // CHECK-NEXT:     %0 = addi
  // CHECK-NEXT:     return
  // CHECK-NEXT: EndLiveness
  %result = addi %arg1, %arg2 : i32
  return %result : i32
}

// -----

// CHECK-LABEL: Testing : func_loop
func @func_loop(%arg0 : i32, %arg1 : i32) -> i32 {
  // CHECK: Block: 0
  // CHECK-NEXT: LiveIn:{{ *$}}
  // CHECK-NEXT: LiveOut: arg1@0
  %const0 = constant 0 : i32
  br ^loopHeader(%const0, %arg0 : i32, i32)
^loopHeader(%counter : i32, %i : i32):
  // CHECK: Block: 1
  // CHECK-NEXT: LiveIn: arg1@0
  // CHECK-NEXT: LiveOut: arg1@0 arg0@1
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: val_std.cmpi
  // CHECK-NEXT:     %2 = cmpi
  // CHECK-NEXT:     cond_br
  // CHECK-NEXT: EndLiveness
  %lessThan = cmpi "slt", %counter, %arg1 : i32
  cond_br %lessThan, ^loopBody(%i : i32), ^exit(%i : i32)
^loopBody(%val : i32):
  // CHECK: Block: 2
  // CHECK-NEXT: LiveIn: arg1@0 arg0@1
  // CHECK-NEXT: LiveOut: arg1@0
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: val_std.constant
  // CHECK-NEXT:     %c
  // CHECK-NEXT:     %4 = addi
  // CHECK-NEXT:     %5 = addi
  // CHECK-NEXT: val_std.addi
  // CHECK-NEXT:     %4 = addi
  // CHECK-NEXT:     %5 = addi
  // CHECK-NEXT:     br
  // CHECK: EndLiveness
  %const1 = constant 1 : i32
  %inc = addi %val, %const1 : i32
  %inc2 = addi %counter, %const1 : i32
  br ^loopHeader(%inc, %inc2 : i32, i32)
^exit(%sum : i32):
  // CHECK: Block: 3
  // CHECK-NEXT: LiveIn: arg1@0
  // CHECK-NEXT: LiveOut:{{ *$}}
  %result = addi %sum, %arg1 : i32
  return %result : i32
}

// -----

// CHECK-LABEL: Testing : func_ranges
func @func_ranges(%cond : i1, %arg1 : i32, %arg2 : i32, %arg3 : i32) -> i32 {
  // CHECK: Block: 0
  // CHECK-NEXT: LiveIn:{{ *$}}
  // CHECK-NEXT: LiveOut: arg2@0 val_std.muli val_std.addi
  // CHECK-NEXT: BeginLiveness
  // CHECK-NEXT: val_std.addi
  // CHECK-NEXT:    %0 = addi
  // CHECK-NEXT:    %c
  // CHECK-NEXT:    %1 = addi
  // CHECK-NEXT:    %2 = addi
  // CHECK-NEXT:    %3 = muli
  // CHECK-NEXT: val_std.constant
  // CHECK-NEXT:    %c
  // CHECK-NEXT:    %1 = addi
  // CHECK-NEXT:    %2 = addi
  // CHECK-NEXT:    %3 = muli
  // CHECK-NEXT:    %4 = muli
  // CHECK-NEXT:    %5 = addi
  // CHECK-NEXT: val_std.addi
  // CHECK-NEXT:    %1 = addi
  // CHECK-NEXT:    %2 = addi
  // CHECK-NEXT:    %3 = muli
  // CHECK-NEXT: val_std.addi
  // CHECK-NEXT    %2 = addi
  // CHECK-NEXT    %3 = muli
  // CHECK-NEXT    %4 = muli
  // CHECK: val_std.muli
  // CHECK-NEXT:    %3 = muli
  // CHECK-NEXT:    %4 = muli
  // CHECK-NEXT: val_std.muli
  // CHECK-NEXT:    %4 = muli
  // CHECK-NEXT:    %5 = addi
  // CHECK-NEXT:    cond_br
  // CHECK-NEXT:    %c
  // CHECK-NEXT:    %6 = muli
  // CHECK-NEXT:    %7 = muli
  // CHECK-NEXT:    %8 = addi
  // CHECK-NEXT: val_std.addi
  // CHECK-NEXT:    %5 = addi
  // CHECK-NEXT:    cond_br
  // CHECK-NEXT:    %7
  // CHECK: EndLiveness
  %0 = addi %arg1, %arg2 : i32
  %const1 = constant 1 : i32
  %1 = addi %const1, %arg2 : i32
  %2 = addi %const1, %arg3 : i32
  %3 = muli %0, %1 : i32
  %4 = muli %3, %2 : i32
  %5 = addi %4, %const1 : i32
  cond_br %cond, ^bb1, ^bb2

^bb1:
  // CHECK: Block: 1
  // CHECK-NEXT: LiveIn: arg2@0 val_std.muli
  // CHECK-NEXT: LiveOut: arg2@0
  %const4 = constant 4 : i32
  %6 = muli %4, %const4 : i32
  br ^exit(%6 : i32)

^bb2:
  // CHECK: Block: 2
  // CHECK-NEXT: LiveIn: arg2@0 val_std.muli val_std.addi
  // CHECK-NEXT: LiveOut: arg2@0
  %7 = muli %4, %5 : i32
  %8 = addi %4, %arg2 : i32
  br ^exit(%8 : i32)

^exit(%sum : i32):
  // CHECK: Block: 3
  // CHECK-NEXT: LiveIn: arg2@0
  // CHECK-NEXT: LiveOut:{{ *$}}
  %result = addi %sum, %arg2 : i32
  return %result : i32
}