// RUN: mlir-opt %s  -convert-scf-to-cf --canonicalize --convert-cf-to-llvm --convert-to-llvm | \
// RUN: mlir-runner -e entry -entry-point-result=void  \
// RUN: -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s



// End-to-end test of all fp reduction intrinsics (not exhaustive unit tests).
module {
  llvm.func @entry() {
   // Constant for the iteration space and various conditions
   %one = llvm.mlir.constant(1 : i64) : i64
   %two = llvm.mlir.constant(2 : i64) : i64
   %three = llvm.mlir.constant(3 : i64) : i64
   %four = llvm.mlir.constant(4 : i64) : i64
   %counter_init = llvm.mlir.constant(0 : i64) : i64


// CHECK: Outer Loop Begin with counter: 0
// CHECK-NEXT: Inner Loop Begin, counter: 1
// CHECK-NEXT: continue inner loop
// CHECK-NEXT: Inner Loop Begin, counter: 2
// CHECK-NEXT: Iteration 2, loop back to outer loop
// CHECK-NEXT: Outer Loop Begin with counter: 2
// CHECK-NEXT: Inner Loop Begin, counter: 3
// CHECK-NEXT: continue inner loop
// CHECK-NEXT: Inner Loop Begin, counter: 4
// CHECK-NEXT: continue inner loop
// CHECK-NEXT: Inner Loop Begin, counter: 5
// CHECK-NEXT: Last iteration, break out of outer loop
// CHECK-NEXT: Outer loop finished with result: 4


   %result = scf.loop iter_args(%counter_out = %counter_init) : i64 -> i64 {
     // Outer loop iteration
     vector.print str "Outer Loop Begin with counter: "
     vector.print %counter_out : i64

     scf.loop iter_args(%counter = %counter_out) : i64 {
       // %counter will go from 0 to 4
       // %counter_update will go from 1 to 5
       %counter_update = llvm.add %counter, %one : i64

       // Inner loop iteration
       // print from 1..5
       vector.print str "Inner Loop Begin, counter: "
       vector.print  %counter_update : i64

       // On the second iteration, print 2.3 and loop back to the outer loop.
       %cond1 = llvm.icmp "eq" %counter_update, %two : i64
       scf.if %cond1 {
         vector.print str "Iteration 2, loop back to outer loop\n"
         scf.continue 3 %counter_update : i64
       }

       // Exit condition when counter>4
       %cond2 = llvm.icmp "sge" %counter, %four : i64
       scf.if %cond2 {
         vector.print str "Last iteration, break out of outer loop\n"
         // return the counter from the previous iteration here (pre-update)
         scf.break 3 %counter : i64
       }

       %cond3 = llvm.icmp "eq" %counter_update, %three : i64
       scf.if %cond2 {
         vector.print str "Iteration 3, break out of inner loop"
         scf.break 2
       }
       vector.print str "continue inner loop\n"
       scf.continue 1 %counter_update : i64
     }
      vector.print str "continue outer loop\n"
      scf.continue 1 %counter_out : i64
   }

// After the loop nest finishes
    vector.print str "Outer loop finished with result: "
    vector.print  %result : i64

    llvm.return
  }
}
