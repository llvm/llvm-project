// RUN: mlir-opt %s -allow-unregistered-dialect -arm-sve-legalize-vector-storage -split-input-file -verify-diagnostics | FileCheck %s

/// This tests the basic functionality of the -arm-sve-legalize-vector-storage pass.

// -----

// CHECK-LABEL: @store_and_reload_sve_predicate_nxv1i1(
// CHECK-SAME:                                         %[[MASK:.*]]: vector<[1]xi1>)
func.func @store_and_reload_sve_predicate_nxv1i1(%mask: vector<[1]xi1>) -> vector<[1]xi1> {
  // CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca() {alignment = 2 : i64} : memref<vector<[16]xi1>>
  %alloca = memref.alloca() : memref<vector<[1]xi1>>
  // CHECK-NEXT: %[[SVBOOL:.*]] = arm_sve.convert_to_svbool %[[MASK]] : vector<[1]xi1>
  // CHECK-NEXT: memref.store %[[SVBOOL]], %[[ALLOCA]][] : memref<vector<[16]xi1>>
  memref.store %mask, %alloca[] : memref<vector<[1]xi1>>
  // CHECK-NEXT: %[[RELOAD:.*]] = memref.load %[[ALLOCA]][] : memref<vector<[16]xi1>>
  // CHECK-NEXT: %[[MASK:.*]] = arm_sve.convert_from_svbool %[[RELOAD]] : vector<[1]xi1>
  %reload = memref.load %alloca[] : memref<vector<[1]xi1>>
  // CHECK-NEXT: return %[[MASK]] : vector<[1]xi1>
  return %reload : vector<[1]xi1>
}

// -----

// CHECK-LABEL: @store_and_reload_sve_predicate_nxv2i1(
// CHECK-SAME:                                         %[[MASK:.*]]: vector<[2]xi1>)
func.func @store_and_reload_sve_predicate_nxv2i1(%mask: vector<[2]xi1>) -> vector<[2]xi1> {
  // CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca() {alignment = 2 : i64} : memref<vector<[16]xi1>>
  %alloca = memref.alloca() : memref<vector<[2]xi1>>
  // CHECK-NEXT: %[[SVBOOL:.*]] = arm_sve.convert_to_svbool %[[MASK]] : vector<[2]xi1>
  // CHECK-NEXT: memref.store %[[SVBOOL]], %[[ALLOCA]][] : memref<vector<[16]xi1>>
  memref.store %mask, %alloca[] : memref<vector<[2]xi1>>
  // CHECK-NEXT: %[[RELOAD:.*]] = memref.load %[[ALLOCA]][] : memref<vector<[16]xi1>>
  // CHECK-NEXT: %[[MASK:.*]] = arm_sve.convert_from_svbool %[[RELOAD]] : vector<[2]xi1>
  %reload = memref.load %alloca[] : memref<vector<[2]xi1>>
  // CHECK-NEXT: return %[[MASK]] : vector<[2]xi1>
  return %reload : vector<[2]xi1>
}

// -----

// CHECK-LABEL: @store_and_reload_sve_predicate_nxv4i1(
// CHECK-SAME:                                         %[[MASK:.*]]: vector<[4]xi1>)
func.func @store_and_reload_sve_predicate_nxv4i1(%mask: vector<[4]xi1>) -> vector<[4]xi1> {
  // CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca() {alignment = 2 : i64} : memref<vector<[16]xi1>>
  %alloca = memref.alloca() : memref<vector<[4]xi1>>
  // CHECK-NEXT: %[[SVBOOL:.*]] = arm_sve.convert_to_svbool %[[MASK]] : vector<[4]xi1>
  // CHECK-NEXT: memref.store %[[SVBOOL]], %[[ALLOCA]][] : memref<vector<[16]xi1>>
  memref.store %mask, %alloca[] : memref<vector<[4]xi1>>
  // CHECK-NEXT: %[[RELOAD:.*]] = memref.load %[[ALLOCA]][] : memref<vector<[16]xi1>>
  // CHECK-NEXT: %[[MASK:.*]] = arm_sve.convert_from_svbool %[[RELOAD]] : vector<[4]xi1>
  %reload = memref.load %alloca[] : memref<vector<[4]xi1>>
  // CHECK-NEXT: return %[[MASK]] : vector<[4]xi1>
  return %reload : vector<[4]xi1>
}

// -----

// CHECK-LABEL: @store_and_reload_sve_predicate_nxv8i1(
// CHECK-SAME:                                         %[[MASK:.*]]: vector<[8]xi1>)
func.func @store_and_reload_sve_predicate_nxv8i1(%mask: vector<[8]xi1>) -> vector<[8]xi1> {
  // CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca() {alignment = 2 : i64} : memref<vector<[16]xi1>>
  %alloca = memref.alloca() : memref<vector<[8]xi1>>
  // CHECK-NEXT: %[[SVBOOL:.*]] = arm_sve.convert_to_svbool %[[MASK]] : vector<[8]xi1>
  // CHECK-NEXT: memref.store %[[SVBOOL]], %[[ALLOCA]][] : memref<vector<[16]xi1>>
  memref.store %mask, %alloca[] : memref<vector<[8]xi1>>
  // CHECK-NEXT: %[[RELOAD:.*]] = memref.load %[[ALLOCA]][] : memref<vector<[16]xi1>>
  // CHECK-NEXT: %[[MASK:.*]] = arm_sve.convert_from_svbool %[[RELOAD]] : vector<[8]xi1>
  %reload = memref.load %alloca[] : memref<vector<[8]xi1>>
  // CHECK-NEXT: return %[[MASK]] : vector<[8]xi1>
  return %reload : vector<[8]xi1>
}

// -----

// CHECK-LABEL: @store_and_reload_sve_predicate_nxv16i1(
// CHECK-SAME:                                         %[[MASK:.*]]: vector<[16]xi1>)
func.func @store_and_reload_sve_predicate_nxv16i1(%mask: vector<[16]xi1>) -> vector<[16]xi1> {
  // CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca() {alignment = 2 : i64} : memref<vector<[16]xi1>>
  %alloca = memref.alloca() : memref<vector<[16]xi1>>
  // CHECK-NEXT: memref.store %[[MASK]], %[[ALLOCA]][] : memref<vector<[16]xi1>>
  memref.store %mask, %alloca[] : memref<vector<[16]xi1>>
  // CHECK-NEXT: %[[RELOAD:.*]] = memref.load %[[ALLOCA]][] : memref<vector<[16]xi1>>
  %reload = memref.load %alloca[] : memref<vector<[16]xi1>>
  // CHECK-NEXT: return %[[RELOAD]] : vector<[16]xi1>
  return %reload : vector<[16]xi1>
}

// -----

/// This is not a valid SVE mask type, so is ignored by the
// `-arm-sve-legalize-vector-storage` pass.

// CHECK-LABEL: @store_and_reload_unsupported_type(
// CHECK-SAME:                                         %[[MASK:.*]]: vector<[7]xi1>)
func.func @store_and_reload_unsupported_type(%mask: vector<[7]xi1>) -> vector<[7]xi1> {
  // CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca() {alignment = 2 : i64} : memref<vector<[7]xi1>>
  %alloca = memref.alloca() : memref<vector<[7]xi1>>
  // CHECK-NEXT: memref.store %[[MASK]], %[[ALLOCA]][] : memref<vector<[7]xi1>>
  memref.store %mask, %alloca[] : memref<vector<[7]xi1>>
  // CHECK-NEXT: %[[RELOAD:.*]] = memref.load %[[ALLOCA]][] : memref<vector<[7]xi1>>
  %reload = memref.load %alloca[] : memref<vector<[7]xi1>>
  // CHECK-NEXT: return %[[RELOAD]] : vector<[7]xi1>
  return %reload : vector<[7]xi1>
}

// -----

// CHECK-LABEL: @store_2d_mask_and_reload_slice(
// CHECK-SAME:                                  %[[MASK:.*]]: vector<3x[8]xi1>)
func.func @store_2d_mask_and_reload_slice(%mask: vector<3x[8]xi1>) -> vector<[8]xi1> {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca() {alignment = 2 : i64} : memref<vector<3x[16]xi1>>
  %alloca = memref.alloca() : memref<vector<3x[8]xi1>>
  // CHECK-NEXT: %[[SVBOOL:.*]] = arm_sve.convert_to_svbool %[[MASK]] : vector<3x[8]xi1>
  // CHECK-NEXT: memref.store %[[SVBOOL]], %[[ALLOCA]][] : memref<vector<3x[16]xi1>>
  memref.store %mask, %alloca[] : memref<vector<3x[8]xi1>>
  // CHECK-NEXT: %[[UNPACK:.*]] = vector.type_cast %[[ALLOCA]] : memref<vector<3x[16]xi1>> to memref<3xvector<[16]xi1>>
  %unpack = vector.type_cast %alloca : memref<vector<3x[8]xi1>> to memref<3xvector<[8]xi1>>
  // CHECK-NEXT: %[[RELOAD:.*]] = memref.load %[[UNPACK]][%[[C0]]] : memref<3xvector<[16]xi1>>
  // CHECK-NEXT: %[[SLICE:.*]] = arm_sve.convert_from_svbool %[[RELOAD]] : vector<[8]xi1>
  %slice = memref.load %unpack[%c0] : memref<3xvector<[8]xi1>>
  // CHECK-NEXT: return %[[SLICE]] : vector<[8]xi1>
  return %slice : vector<[8]xi1>
}

// -----

// CHECK-LABEL: @set_sve_alloca_alignment
func.func @set_sve_alloca_alignment() {
  /// This checks the alignment of alloca's of scalable vectors will be
  /// something the backend can handle. Currently, the backend sets the
  /// alignment of scalable vectors to their base size (i.e. their size at
  /// vscale = 1). This works for hardware-sized types, which always get a
  /// 16-byte alignment. The problem is larger types e.g. vector<[8]xf32> end up
  /// with alignments larger than 16-bytes (e.g. 32-bytes here), which are
  /// unsupported. The `-arm-sve-legalize-vector-storage` pass avoids this
  /// issue by explicitly setting the alignment to 16-bytes for all scalable
  /// vectors.

  // CHECK-COUNT-6: alignment = 16
  %a1 = memref.alloca() : memref<vector<[32]xi8>>
  %a2 = memref.alloca() : memref<vector<[16]xi8>>
  %a3 = memref.alloca() : memref<vector<[8]xi8>>
  %a4 = memref.alloca() : memref<vector<[4]xi8>>
  %a5 = memref.alloca() : memref<vector<[2]xi8>>
  %a6 = memref.alloca() : memref<vector<[1]xi8>>

  // CHECK-COUNT-6: alignment = 16
  %b1 = memref.alloca() : memref<vector<[32]xi16>>
  %b2 = memref.alloca() : memref<vector<[16]xi16>>
  %b3 = memref.alloca() : memref<vector<[8]xi16>>
  %b4 = memref.alloca() : memref<vector<[4]xi16>>
  %b5 = memref.alloca() : memref<vector<[2]xi16>>
  %b6 = memref.alloca() : memref<vector<[1]xi16>>

  // CHECK-COUNT-6: alignment = 16
  %c1 = memref.alloca() : memref<vector<[32]xi32>>
  %c2 = memref.alloca() : memref<vector<[16]xi32>>
  %c3 = memref.alloca() : memref<vector<[8]xi32>>
  %c4 = memref.alloca() : memref<vector<[4]xi32>>
  %c5 = memref.alloca() : memref<vector<[2]xi32>>
  %c6 = memref.alloca() : memref<vector<[1]xi32>>

  // CHECK-COUNT-6: alignment = 16
  %d1 = memref.alloca() : memref<vector<[32]xi64>>
  %d2 = memref.alloca() : memref<vector<[16]xi64>>
  %d3 = memref.alloca() : memref<vector<[8]xi64>>
  %d4 = memref.alloca() : memref<vector<[4]xi64>>
  %d5 = memref.alloca() : memref<vector<[2]xi64>>
  %d6 = memref.alloca() : memref<vector<[1]xi64>>

  // CHECK-COUNT-6: alignment = 16
  %e1 = memref.alloca() : memref<vector<[32]xf32>>
  %e2 = memref.alloca() : memref<vector<[16]xf32>>
  %e3 = memref.alloca() : memref<vector<[8]xf32>>
  %e4 = memref.alloca() : memref<vector<[4]xf32>>
  %e5 = memref.alloca() : memref<vector<[2]xf32>>
  %e6 = memref.alloca() : memref<vector<[1]xf32>>

  // CHECK-COUNT-6: alignment = 16
  %f1 = memref.alloca() : memref<vector<[32]xf64>>
  %f2 = memref.alloca() : memref<vector<[16]xf64>>
  %f3 = memref.alloca() : memref<vector<[8]xf64>>
  %f4 = memref.alloca() : memref<vector<[4]xf64>>
  %f5 = memref.alloca() : memref<vector<[2]xf64>>
  %f6 = memref.alloca() : memref<vector<[1]xf64>>

  "prevent.dce"(
    %a1, %a2, %a3, %a4, %a5, %a6,
    %b1, %b2, %b3, %b4, %b5, %b6,
    %c1, %c2, %c3, %c4, %c5, %c6,
    %d1, %d2, %d3, %d4, %d5, %d6,
    %e1, %e2, %e3, %e4, %e5, %e6,
    %f1, %f2, %f3, %f4, %f5, %f6)
    : (memref<vector<[32]xi8>>, memref<vector<[16]xi8>>, memref<vector<[8]xi8>>, memref<vector<[4]xi8>>, memref<vector<[2]xi8>>, memref<vector<[1]xi8>>,
       memref<vector<[32]xi16>>, memref<vector<[16]xi16>>, memref<vector<[8]xi16>>, memref<vector<[4]xi16>>, memref<vector<[2]xi16>>, memref<vector<[1]xi16>>,
       memref<vector<[32]xi32>>, memref<vector<[16]xi32>>, memref<vector<[8]xi32>>, memref<vector<[4]xi32>>, memref<vector<[2]xi32>>, memref<vector<[1]xi32>>,
       memref<vector<[32]xi64>>, memref<vector<[16]xi64>>, memref<vector<[8]xi64>>, memref<vector<[4]xi64>>, memref<vector<[2]xi64>>, memref<vector<[1]xi64>>,
       memref<vector<[32]xf32>>, memref<vector<[16]xf32>>, memref<vector<[8]xf32>>, memref<vector<[4]xf32>>, memref<vector<[2]xf32>>, memref<vector<[1]xf32>>,
       memref<vector<[32]xf64>>, memref<vector<[16]xf64>>, memref<vector<[8]xf64>>, memref<vector<[4]xf64>>, memref<vector<[2]xf64>>, memref<vector<[1]xf64>>) -> ()
  return
}
