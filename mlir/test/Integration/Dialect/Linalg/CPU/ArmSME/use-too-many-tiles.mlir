// RUN: mlir-opt %s \
// RUN:   -convert-vector-to-arm-sme -convert-arith-to-arm-sme \
// RUN:   -allocate-arm-sme-tiles -convert-arm-sme-to-scf \
// RUN:   -enable-arm-streaming="streaming-mode=streaming-locally za-mode=new-za only-if-required-by-ops"  \
// RUN:   -convert-vector-to-scf -cse -arm-sve-legalize-vector-storage \
// RUN:   -convert-arm-sme-to-llvm -convert-vector-to-llvm=enable-arm-sve -cse \
// RUN:   -canonicalize -test-lower-to-llvm -verify-diagnostics | \
// RUN: %mcr_aarch64_cmd \
// RUN:   -e=main -entry-point-result=void \
// RUN:   -march=aarch64 -mattr="+sve,+sme" \
// RUN:   -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%arm_sme_abi_shlib | \
// RUN: FileCheck %s

/// This function uses too many tiles! There's only two i16 tiles (ZA0.H and
/// ZA1.H), but this function uses five i16 tiles! Very expensive spills/reloads
/// will be inserted to emulate the extra three tiles. Note: This is only done
/// to avoid the compiler erroring out but is expected to have very poor
/// performance (hence the warning).
func.func @use_too_many_tiles(%a: memref<?x?xi16>, %b:  memref<?x?xi16>, %c: memref<?x?xi16>) {
  %c0 = arith.constant 0 : index
  %tile_a = arith.constant dense<0> : vector<[8]x[8]xi16>
  %tile_b = arith.constant dense<1> : vector<[8]x[8]xi16>
  // expected-warning @below {{failed to allocate SME virtual tile to operation, all tile operations will go through memory, expect degraded performance}}
  %tile_c = arm_sme.tile_load %a[%c0, %c0] : memref<?x?xi16>, vector<[8]x[8]xi16>
  // expected-warning @below {{failed to allocate SME virtual tile to operation, all tile operations will go through memory, expect degraded performance}}
  %tile_d = arm_sme.tile_load %b[%c0, %c0] : memref<?x?xi16>, vector<[8]x[8]xi16>
  // expected-warning @below {{failed to allocate SME virtual tile to operation, all tile operations will go through memory, expect degraded performance}}
  %tile_e = arm_sme.tile_load %c[%c0, %c0] : memref<?x?xi16>, vector<[8]x[8]xi16>

  // CHECK-LABEL: tile_a:
  // CHECK-COUNT-8: ( 0, 0, 0, 0, 0, 0, 0, 0
  vector.print str "tile_a:"
  vector.print %tile_a : vector<[8]x[8]xi16>
  // CHECK-LABEL: tile_b:
  // CHECK-COUNT-8: ( 1, 1, 1, 1, 1, 1, 1, 1
  vector.print str "tile_b:"
  vector.print %tile_b : vector<[8]x[8]xi16>
  // CHECK-LABEL: tile_c:
  // CHECK-COUNT-8: ( 2, 2, 2, 2, 2, 2, 2, 2
  vector.print str "tile_c:"
  vector.print %tile_c : vector<[8]x[8]xi16>
  // CHECK-LABEL: tile_d:
  // CHECK-COUNT-8: ( 3, 3, 3, 3, 3, 3, 3, 3
  vector.print str "tile_d:"
  vector.print %tile_d : vector<[8]x[8]xi16>
  // CHECK-LABEL: tile_e:
  // CHECK-COUNT-8: ( 4, 4, 4, 4, 4, 4, 4, 4
  vector.print str "tile_e:"
  vector.print %tile_e : vector<[8]x[8]xi16>
  return
}

func.func @main() {
  %c16 = arith.constant 16 : index
  %svl_h = arm_sme.streaming_vl <half>

  %c2 = arith.constant 2 : i16
  %c3 = arith.constant 3 : i16
  %c4 = arith.constant 4 : i16

  %memA = memref.alloca(%svl_h, %svl_h) : memref<?x?xi16>
  %memB = memref.alloca(%svl_h, %svl_h) : memref<?x?xi16>
  %memC = memref.alloca(%svl_h, %svl_h) : memref<?x?xi16>

  linalg.fill ins(%c2 : i16) outs(%memA : memref<?x?xi16>)
  linalg.fill ins(%c3 : i16) outs(%memB : memref<?x?xi16>)
  linalg.fill ins(%c4 : i16) outs(%memC : memref<?x?xi16>)

  func.call @use_too_many_tiles(%memA, %memB, %memC) : (memref<?x?xi16>, memref<?x?xi16>, memref<?x?xi16>) -> ()
  return
}
