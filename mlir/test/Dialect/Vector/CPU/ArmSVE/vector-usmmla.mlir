// RUN:  mlir-opt %s --convert-vector-to-llvm='enable-arm-sve enable-arm-i8mm' --split-input-file | FileCheck %s

#packed_maps = [
  affine_map<(d0, d1, d2) -> (d0, d2)>,
  affine_map<(d0, d1, d2) -> (d1, d2)>,
  affine_map<(d0, d1, d2) -> (d0, d1)>
]

// CHECK-LABEL: @test_vector_contract_to_usmmla

// Extract LHS rows 0 and 1, concatenate, turn into scalable vector
// CHECK: %[[T6:[0-9]+]] = llvm.extractvalue %[[T4:[0-9]+]][0] : !llvm.array<4 x vector<8xi8>>
// CHECK-NEXT: %[[T7:[0-9]+]] = llvm.extractvalue %[[T4]][1] : !llvm.array<4 x vector<8xi8>>
// CHECK-NEXT: %[[T8:[0-9]+]] = llvm.shufflevector %[[T6]], %[[T7]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xi8>
// CHECK-NEXT: %[[T9:[0-9]+]] = llvm.intr.vector.insert %[[T8]], %[[T0:[0-9+]]][0] : vector<16xi8> into vector<[16]xi8>

// Replicate across the entire length of the scalabale vector
// CHECK-NEXT: %[[T10:[0-9]+]] = "arm_sve.intr.dupq_lane"(%[[T9]]) <{lane = 0 : i64}> : (vector<[16]xi8>) -> vector<[16]xi8>

// Same for LHS rows 2 and 4
// CHECK-NEXT: %[[T11:[0-9]+]] = llvm.extractvalue %[[T4]][2] : !llvm.array<4 x vector<8xi8>>
// CHECK-NEXT: %[[T12:[0-9]+]] = llvm.extractvalue %[[T4]][3] : !llvm.array<4 x vector<8xi8>>
// CHECK-NEXT: %[[T13:[0-9]+]] = llvm.shufflevector %[[T11]], %[[T12]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xi8>
// CHECK-NEXT: %[[T14:[0-9]+]] = llvm.intr.vector.insert %[[T13]], %[[T0]][0] : vector<16xi8> into vector<[16]xi8>
// CHECK-NEXT: %[[T15:[0-9]+]] = "arm_sve.intr.dupq_lane"(%[[T14]]) <{lane = 0 : i64}> : (vector<[16]xi8>) -> vector<[16]xi8>

// Extract sub-tiles from the RHS
// CHECK-NEXT: %[[T16:[0-9]+]] = vector.shape_cast %arg1 : vector<[4]x8xi8> to vector<[32]xi8>
// CHECK-NEXT: %[[T17:[0-9]+]] = llvm.intr.vector.extract %[[T16]][0] : vector<[16]xi8> from vector<[32]xi8>
// CHECK-NEXT: %[[T18:[0-9]+]] = llvm.intr.vector.extract %[[T16]][16] : vector<[16]xi8> from vector<[32]xi8>

// Extract accumulator rows 0 and 1 and pack (into "registers")
// CHECK-NEXT: %[[T19:[0-9]+]] = llvm.extractvalue %[[T3:[0-9]+]][0] : !llvm.array<4 x vector<[4]xi32>>
// CHECK-NEXT: %[[T20:[0-9]+]] = llvm.extractvalue %[[T3]][1] : !llvm.array<4 x vector<[4]xi32>>
// CHECK-NEXT: %[[T21:[0-9]+]] = llvm.bitcast %[[T19]] : vector<[4]xi32> to vector<[2]xi64>
// CHECK-NEXT: %[[T22:[0-9]+]] = llvm.bitcast %[[T20]] : vector<[4]xi32> to vector<[2]xi64>
// CHECK-NEXT: %[[T23:[0-9]+]] = "llvm.intr.vector.interleave2"(%[[T21]], %[[T22]]) : (vector<[2]xi64>, vector<[2]xi64>) -> vector<[4]xi64>
// CHECK-NEXT: %[[T24:[0-9]+]] = llvm.bitcast %[[T23]] : vector<[4]xi64> to vector<[8]xi32>
// CHECK-NEXT: %[[T25:[0-9]+]] = llvm.intr.vector.extract %[[T24]][0] : vector<[4]xi32> from vector<[8]xi32>
// CHECK-NEXT: %[[T26:[0-9]+]] = llvm.intr.vector.extract %[[T24]][4] : vector<[4]xi32> from vector<[8]xi32>

// Same for accumulator rows 2 and 3.
// CHECK-NEXT: %[[T27:[0-9]+]] = llvm.extractvalue %[[T3]][2] : !llvm.array<4 x vector<[4]xi32>>
// CHECK-NEXT: %[[T28:[0-9]+]] = llvm.extractvalue %[[T3]][3] : !llvm.array<4 x vector<[4]xi32>>
// CHECK-NEXT: %[[T29:[0-9]+]] = llvm.bitcast %[[T27]] : vector<[4]xi32> to vector<[2]xi64>
// CHECK-NEXT: %[[T30:[0-9]+]] = llvm.bitcast %[[T28]] : vector<[4]xi32> to vector<[2]xi64>
// CHECK-NEXT: %[[T31:[0-9]+]] = "llvm.intr.vector.interleave2"(%[[T29]], %[[T30]]) : (vector<[2]xi64>, vector<[2]xi64>) -> vector<[4]xi64>
// CHECK-NEXT: %[[T32:[0-9]+]] = llvm.bitcast %[[T31]] : vector<[4]xi64> to vector<[8]xi32>
// CHECK-NEXT: %[[T33:[0-9]+]] = llvm.intr.vector.extract %[[T32]][0] : vector<[4]xi32> from vector<[8]xi32>
// CHECK-NEXT: %[[T34:[0-9]+]] = llvm.intr.vector.extract %[[T32]][4] : vector<[4]xi32> from vector<[8]xi32>

// Do the sub-tile matrix multiplications
// CHECK-NEXT: %[[T35:[0-9]+]] = "arm_sve.intr.usmmla"(%[[T25]], %[[T10]], %[[T17]]) : (vector<[4]xi32>, vector<[16]xi8>, vector<[16]xi8>) -> vector<[4]xi32>
// CHECK-NEXT: %[[T36:[0-9]+]] = "arm_sve.intr.usmmla"(%[[T26]], %[[T10]], %[[T18]]) : (vector<[4]xi32>, vector<[16]xi8>, vector<[16]xi8>) -> vector<[4]xi32>
// CHECK-NEXT: %[[T37:[0-9]+]] = "arm_sve.intr.usmmla"(%[[T33]], %[[T15]], %[[T17]]) : (vector<[4]xi32>, vector<[16]xi8>, vector<[16]xi8>) -> vector<[4]xi32>
// CHECK-NEXT: %[[T38:[0-9]+]] = "arm_sve.intr.usmmla"(%[[T34]], %[[T15]], %[[T18]]) : (vector<[4]xi32>, vector<[16]xi8>, vector<[16]xi8>) -> vector<[4]xi32>

// Unpack (from "registers") and insert in the output result rows  0 and 1
// CHECK-NEXT: %[[T39:[0-9]+]] = llvm.intr.vector.insert %[[T35]], %[[T2:[0-9]+]][0] : vector<[4]xi32> into vector<[8]xi32>
// CHECK-NEXT: %[[T40:[0-9]+]] = llvm.intr.vector.insert %[[T36]], %[[T39]][4] : vector<[4]xi32> into vector<[8]xi32>
// CHECK-NEXT: %[[T41:[0-9]+]] = llvm.bitcast %[[T40]] : vector<[8]xi32> to vector<[4]xi64>
// CHECK-NEXT: %[[T42:[0-9]+]] = "llvm.intr.vector.deinterleave2"(%[[T41]]) : (vector<[4]xi64>) -> !llvm.struct<(vector<[2]xi64>, vector<[2]xi64>)>
// CHECK-NEXT: %[[T43:[0-9]+]] = llvm.extractvalue %[[T42]][0] : !llvm.struct<(vector<[2]xi64>, vector<[2]xi64>)>
// CHECK-NEXT: %[[T44:[0-9]+]] = llvm.extractvalue %[[T42]][1] : !llvm.struct<(vector<[2]xi64>, vector<[2]xi64>)>
// CHECK-NEXT: %[[T45:[0-9]+]] = llvm.bitcast %[[T43]] : vector<[2]xi64> to vector<[4]xi32>
// CHECK-NEXT: %[[T46:[0-9]+]] = llvm.bitcast %[[T44]] : vector<[2]xi64> to vector<[4]xi32>
// CHECK-NEXT: %[[T47:[0-9]+]] = llvm.insertvalue %[[T45]], %[[T5:[0-9]+]][0] : !llvm.array<4 x vector<[4]xi32>>
// CHECK-NEXT: %[[T48:[0-9]+]] = llvm.insertvalue %[[T46]], %[[T47]][1] : !llvm.array<4 x vector<[4]xi32>>

// Same for result rows 2 and 3
// CHECK-NEXT: %[[T49:[0-9]+]] = llvm.intr.vector.insert %[[T37]], %[[T2]][0] : vector<[4]xi32> into vector<[8]xi32>
// CHECK-NEXT: %[[T50:[0-9]+]] = llvm.intr.vector.insert %[[T38]], %[[T49]][4] : vector<[4]xi32> into vector<[8]xi32>
// CHECK-NEXT: %[[T51:[0-9]+]] = llvm.bitcast %[[T50]] : vector<[8]xi32> to vector<[4]xi64>
// CHECK-NEXT: %[[T52:[0-9]+]] = "llvm.intr.vector.deinterleave2"(%[[T51]]) : (vector<[4]xi64>) -> !llvm.struct<(vector<[2]xi64>, vector<[2]xi64>)>
// CHECK-NEXT: %[[T53:[0-9]+]] = llvm.extractvalue %[[T52]][0] : !llvm.struct<(vector<[2]xi64>, vector<[2]xi64>)>
// CHECK-NEXT: %[[T54:[0-9]+]] = llvm.extractvalue %[[T52]][1] : !llvm.struct<(vector<[2]xi64>, vector<[2]xi64>)>
// CHECK-NEXT: %[[T55:[0-9]+]] = llvm.bitcast %[[T53]] : vector<[2]xi64> to vector<[4]xi32>
// CHECK-NEXT: %[[T56:[0-9]+]] = llvm.bitcast %[[T54]] : vector<[2]xi64> to vector<[4]xi32>
// CHECK-NEXT: %[[T57:[0-9]+]] = llvm.insertvalue %[[T55]], %[[T48]][2] : !llvm.array<4 x vector<[4]xi32>>
// CHECK-NEXT: %[[T58:[0-9]+]] = llvm.insertvalue %[[T56]], %[[T57]][3] : !llvm.array<4 x vector<[4]xi32>>

func.func @test_vector_contract_to_usmmla(
  %lhs: vector<4x8xi8>,
  %rhs: vector<[4]x8xi8>,
  %acc: vector<4x[4]xi32>) -> vector<4x[4]xi32> {

  %0 = arith.extui %lhs : vector<4x8xi8> to vector<4x8xi32>
  %1 = arith.extsi %rhs : vector<[4]x8xi8> to vector<[4]x8xi32>
  %2 = vector.contract {indexing_maps = #packed_maps,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %acc
    : vector<4x8xi32>, vector<[4]x8xi32> into vector<4x[4]xi32>

  return %2 : vector<4x[4]xi32>
}
