// RUN: mlir-opt %s -test-liveness-analysis -split-input-file 2>&1 | FileCheck %s --check-prefix=LIVE
// RUN: mlir-opt %s -test-next-access -split-input-file | FileCheck %s --check-prefix=ACCESS

// LIVE-LABEL: test_tag: dead:
// LIVE: operand #0: live
// LIVE: operand #1: not live
// LIVE: result #0: not live
// LIVE-LABEL: test_tag: live:
// LIVE: operand #0: live
// LIVE: operand #1: live
// LIVE: result #0: live
func.func @liveness(%cond : i1, %x : i32, %y : i32) -> i32 {
  %0 = scf.loop token(%loop) iter_args(%a = %x) : i32 -> i32 {
    %dead = arith.addi %a, %y {tag = "dead"} : i32
    %live = arith.addi %a, %a {tag = "live"} : i32
    scf.if %cond {
      scf.break [%loop] %live : i32
    }
    scf.continue [%loop] %a : i32
  }
  return %0 : i32
}

// -----

// ACCESS-LABEL: func.func @next_access
// The first entry is for the scalar store value, whose next memory access is
// unknown. The memref's next access is either the inner load or the loop store.
// ACCESS: name = "before", next_access = ["unknown", ["inloop", "inif"]]
// ACCESS: name = "inif", next_access = ["unknown"]
// ACCESS: name = "inloop", next_access = ["unknown", "unknown"]
// ACCESS: name = "after", next_access = ["unknown"]
func.func @next_access(%cond : i1, %m : memref<i32>) {
  %c = arith.constant 0 : i32
  memref.store %c, %m[] {name = "before"} : memref<i32>
  scf.loop token(%loop) {
    scf.if %cond {
      memref.load %m[] {name = "inif"} : memref<i32>
      scf.break [%loop]
    }
    memref.store %c, %m[] {name = "inloop"} : memref<i32>
    scf.continue [%loop]
  }
  memref.load %m[] {name = "after"} : memref<i32>
  return
}
