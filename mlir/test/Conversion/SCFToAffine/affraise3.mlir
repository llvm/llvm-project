// RUN: mlir-opt --affine-cfg --raise-scf-to-affine %s | FileCheck %s

module {
  func.func @slt(%arg0: index) {
    affine.for %arg1 = 0 to 10 {
      %c = arith.cmpi slt, %arg1, %arg0 : index
      scf.if %c {
        "test.run"(%arg1) : (index) -> ()
      }
    }
    return
  }
  func.func @sle(%arg0: index) {
    affine.for %arg1 = 0 to 10 {
      %c = arith.cmpi sle, %arg1, %arg0 : index
      scf.if %c {
        "test.run"(%arg1) : (index) -> ()
      }
    }
    return
  }
  func.func @sgt(%arg0: index) {
    affine.for %arg1 = 0 to 10 {
      %c = arith.cmpi sgt, %arg1, %arg0 : index
      scf.if %c {
        "test.run"(%arg1) : (index) -> ()
      }
    }
    return
  }
  func.func @sge(%arg0: index) {
    affine.for %arg1 = 0 to 10 {
      %c = arith.cmpi sge, %arg1, %arg0 : index
      scf.if %c {
        "test.run"(%arg1) : (index) -> ()
      }
    }
    return
  }
}

//  -d0 + s0 - 1 >= 0  =>
//  -d0 >= 1 - s0 
//  d0 <= s0 - 1 
//  d0 < s0 
// CHECK: #set = affine_set<(d0)[s0] : (-d0 + s0 - 1 >= 0)>


//  -d0 + s0  >= 0  =>
//  -d0 >= - s0 
//  d0 <= s0
// CHECK: #set1 = affine_set<(d0)[s0] : (-d0 + s0 >= 0)>

//  d0 - s0 - 1 >= 0  =>
//  d0 >= s0 + 1 
//  d0 > s0
// CHECK: #set2 = affine_set<(d0)[s0] : (d0 - s0 - 1 >= 0)>

//  d0 - s0 >= 0  =>
//  d0 >= s0 
// CHECK: #set3 = affine_set<(d0)[s0] : (d0 - s0 >= 0)>

// CHECK:   func.func @slt(%[[arg0:.+]]: index) {
// CHECK-NEXT:     affine.for %[[arg1:.+]] = 0 to 10 {
// CHECK-NEXT:       affine.if #set(%arg1)[%[[arg0]]] {
// CHECK-NEXT:         "test.run"(%[[arg1]]) : (index) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @sle(%[[arg0:.+]]: index) {
// CHECK-NEXT:     affine.for %[[arg1:.+]] = 0 to 10 {
// CHECK-NEXT:       affine.if #set1(%arg1)[%[[arg0]]] {
// CHECK-NEXT:         "test.run"(%[[arg1]]) : (index) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @sgt(%[[arg0:.+]]: index) {
// CHECK-NEXT:     affine.for %[[arg1:.+]] = 0 to 10 {
// CHECK-NEXT:       affine.if #set2(%arg1)[%[[arg0]]] {
// CHECK-NEXT:         "test.run"(%[[arg1]]) : (index) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @sge(%[[arg0:.+]]: index) {
// CHECK-NEXT:     affine.for %[[arg1:.+]] = 0 to 10 {
// CHECK-NEXT:       affine.if #set3(%arg1)[%[[arg0]]] {
// CHECK-NEXT:         "test.run"(%[[arg1]]) : (index) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

