// RUN: mlir-opt --affine-cfg --split-input-file %s | FileCheck %s

module {
  func.func @_Z7runTestiPPc(%arg0: index, %arg2: memref<?xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %1 = arith.addi %arg0, %c1 : index
    affine.for %arg3 = 0 to 2 {
      %2 = arith.muli %arg3, %1 : index
      affine.for %arg4 = 0 to 2 {
        %3 = arith.addi %2, %arg4 : index
        memref.store %c0_i32, %arg2[%3] : memref<?xi32>
      }
    }
    return
  }

}


// CHECK:   func.func @_Z7runTestiPPc(%[[arg0:.+]]: index, %[[arg1:.+]]: memref<?xi32>) {
// CHECK-NEXT:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:     affine.for %[[arg2:.+]] = 0 to 2 {
// CHECK-NEXT:       affine.for %[[arg3:.+]] = 0 to 2 {
// CHECK-NEXT:         affine.store %c0_i32, %arg1[%[[arg3]] + %[[arg2]] * (symbol(%[[arg0]]) + 1)] : memref<?xi32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// -----
module {
func.func @kernel_nussinov(%arg0: i32, %arg2: memref<i32>) {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  %c1_i32 = arith.constant 1 : i32
  %c59 = arith.constant 59 : index
  %c100_i32 = arith.constant 100 : i32
  affine.for %arg3 = 0 to 60 {
    %0 = arith.subi %c59, %arg3 : index
    %1 = arith.index_cast %0 : index to i32
    %2 = arith.cmpi slt, %1, %c100_i32 : i32
    scf.if %2 {
      affine.store %arg0, %arg2[] : memref<i32>
    }
  }
  return
}
}

// CHECK: #set = affine_set<(d0) : (d0 + 40 >= 0)>
// CHECK:   func.func @kernel_nussinov(%[[arg0:.+]]: i32, %[[arg1:.+]]: memref<i32>) {
// CHECK-NEXT:     affine.for %[[arg2:.+]] = 0 to 60 {
// CHECK-NEXT:       affine.if #set(%[[arg2]]) {
// CHECK-NEXT:         affine.store %[[arg0]], %[[arg1]][] : memref<i32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }


// -----

module {
  func.func private @run()

  func.func @minif(%arg4: i32, %arg5 : i32, %arg10 : index) { 
    %c0_i32 = arith.constant 0 : i32
      
    affine.for %i = 0 to 10 {
      %70 = arith.index_cast %arg10 : index to i32
      %71 = arith.muli %70, %arg5 : i32
      %73 = arith.divui %71, %arg5 : i32
      %75 = arith.muli %73, %arg5 : i32
      %79 = arith.subi %arg4, %75 : i32
      %81 = arith.cmpi sle, %arg5, %79 : i32
      %83 = arith.select %81, %arg5, %79 : i32
      %92 = arith.cmpi slt, %c0_i32, %83 : i32
      scf.if %92 {
        func.call @run() : () -> ()
        scf.yield
      }
    }
    return
  }
}

// CHECK: #set = affine_set<()[s0] : (s0 - 1 >= 0)>
// CHECK:   func.func @minif(%[[arg0:.+]]: i32, %[[arg1:.+]]: i32, %[[arg2:.+]]: index) {
// CHECK-NEXT:     %[[V0:.+]] = arith.index_cast %[[arg2]] : index to i32
// CHECK-NEXT:     %[[V1:.+]] = arith.muli %[[V0]], %[[arg1]] : i32
// CHECK-NEXT:     %[[V2:.+]] = arith.divui %[[V1]], %[[arg1]] : i32
// CHECK-NEXT:     %[[V3:.+]] = arith.muli %[[V2]], %[[arg1]] : i32
// CHECK-NEXT:     %[[V4:.+]] = arith.subi %[[arg0]], %[[V3]] : i32
// CHECK-NEXT:     %[[V5:.+]] = arith.cmpi sle, %[[arg1]], %[[V4]] : i32
// CHECK-NEXT:     %[[V6:.+]] = arith.select %5, %[[arg1]], %[[V4]] : i32
// CHECK-NEXT:     %[[V7:.+]] = arith.index_cast %[[V6]] : i32 to index
// CHECK-NEXT:     affine.for %[[arg3:.+]] = 0 to 10 {
// CHECK-NEXT:       affine.if #set()[%[[V7]]] {
// CHECK-NEXT:         func.call @run() : () -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// -----

module {
  llvm.func @atoi(!llvm.ptr) -> i32
func.func @_Z7runTestiPPc(%arg0: i32, %39: memref<?xi32>, %arg1: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<external>} {
  %c2_i32 = arith.constant 2 : i32
  %c16_i32 = arith.constant 16 : i32
    %58 = llvm.call @atoi(%arg1) : (!llvm.ptr) -> i32
  %40 = arith.divsi %58, %c16_i32 : i32
  affine.for %arg2 = 1 to 10 {
      %62 = arith.index_cast %arg2 : index to i32
      %67 = arith.muli %58, %62 : i32
      %69 = arith.addi %67, %40 : i32
        %75 = arith.addi %69, %58 : i32
        %76 = arith.index_cast %75 : i32 to index
        memref.store %c2_i32, %39[%76] : memref<?xi32>
  }
  return
}
}

// CHECK:   func.func @_Z7runTestiPPc(%[[arg0:.+]]: i32, %[[arg1:.+]]: memref<?xi32>, %[[arg2:.+]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[c2_i32:.+]] = arith.constant 2 : i32
// CHECK-NEXT:     %[[c16_i32:.+]] = arith.constant 16 : i32
// CHECK-NEXT:     %[[V0:.+]] = llvm.call @atoi(%[[arg2]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:     %[[V1:.+]] = arith.index_cast %[[V0]] : i32 to index
// CHECK-NEXT:     %[[V2:.+]] = arith.divsi %[[V0]], %[[c16_i32]] : i32
// CHECK-NEXT:     %[[V3:.+]] = arith.index_cast %[[V2]] : i32 to index
// CHECK-NEXT:     affine.for %[[arg3:.+]] = 1 to 10 {
// CHECK-NEXT:       affine.store %[[c2_i32]], %[[arg1]][%[[arg3]] * symbol(%1) + symbol(%1) + symbol(%[[V3]])] : memref<?xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// -----

module {
  func.func @c(%71: memref<?xf32>, %39: i64) {
      affine.parallel (%arg2, %arg3) = (0, 0) to (42, 512) {
        %262 = arith.index_cast %arg2 : index to i32
        %a264 = arith.extsi %262 : i32 to i64
        %268 = arith.cmpi slt, %a264, %39 : i64
        scf.if %268 {
          "test.something"() : () -> ()
        }
      }
    return
  }
}

// CHECK: #set = affine_set<(d0)[s0] : (-d0 + s0 - 1 >= 0)>
// CHECK:   func.func @c(%[[arg0:.+]]: memref<?xf32>, %[[arg1]]: i64) {
// CHECK-NEXT:     %[[V0:.+]] = arith.index_cast %[[arg1]] : i64 to index
// CHECK-NEXT:     affine.parallel (%[[arg2:.+]], %[[arg3:.+]]) = (0, 0) to (42, 512) {
// CHECK-NEXT:       affine.if #set(%[[arg2]])[%[[V0]]] {
// CHECK-NEXT:         "test.something"() : () -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

