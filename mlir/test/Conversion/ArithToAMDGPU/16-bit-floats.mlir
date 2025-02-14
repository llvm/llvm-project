// RUN: mlir-opt --split-input-file %s -convert-arith-to-amdgpu="allow-packed-f16-round-to-zero=true" | FileCheck %s

// CHECK-LABEL: @scalar_trunc
// CHECK-SAME: (%[[value:.*]]: f32)
func.func @scalar_trunc(%v: f32) -> f16{
  // CHECK: %[[poison:.*]] = llvm.mlir.poison : f32
  // CHECK: %[[trunc:.*]] = rocdl.cvt.pkrtz %[[value]], %[[poison]] : vector<2xf16>
  // CHECK: %[[extract:.*]] = vector.extract %[[trunc]][0] : f16 from vector<2xf16>
  // CHECK: return %[[extract]] : f16
  %w = arith.truncf %v : f32 to f16
  return %w : f16
}

// CHECK-LABEL: @vector_trunc
// CHECK-SAME: (%[[value:.*]]: vector<2xf32>)
func.func @vector_trunc_short(%v: vector<2xf32>) -> vector<2xf16> {
  // CHECK: %[[elem0:.*]] = vector.extract %[[value]]
  // CHECK: %[[elem1:.*]] = vector.extract %[[value]]
  // CHECK: %[[ret:.*]] = rocdl.cvt.pkrtz %[[elem0]], %[[elem1]] : vector<2xf16>
  // CHECK: return %[[ret]]
  %w = arith.truncf %v : vector<2xf32> to vector<2xf16>
  return %w : vector<2xf16>
}

// CHECK-LABEL:  @vector_trunc_long
// CHECK-SAME: (%[[value:.*]]: vector<9xf32>)
func.func @vector_trunc_long(%v: vector<9xf32>) -> vector<9xf16> {
  // CHECK: %[[elem0:.*]] = vector.extract %[[value]][0]
  // CHECK: %[[elem1:.*]] = vector.extract %[[value]][1]
  // CHECK: %[[packed0:.*]] = rocdl.cvt.pkrtz %[[elem0]], %[[elem1]] : vector<2xf16>
  // CHECK: %[[out0:.*]] = vector.insert_strided_slice %[[packed0]], {{.*}} {offsets = [0], strides = [1]} : vector<2xf16> into vector<9xf16>
  // CHECK: %[[elem2:.*]] = vector.extract %[[value]][2]
  // CHECK: %[[elem3:.*]] = vector.extract %[[value]][3]
  // CHECK: %[[packed1:.*]] = rocdl.cvt.pkrtz %[[elem2]], %[[elem3]] : vector<2xf16>
  // CHECK: %[[out1:.*]] = vector.insert_strided_slice %[[packed1]], %[[out0]] {offsets = [2], strides = [1]} : vector<2xf16> into vector<9xf16>
  // CHECK: %[[elem4:.*]] = vector.extract %[[value]][4]
  // CHECK: %[[elem5:.*]] = vector.extract %[[value]][5]
  // CHECK: %[[packed2:.*]] = rocdl.cvt.pkrtz %[[elem4]], %[[elem5]] : vector<2xf16>
  // CHECK: %[[out2:.*]] = vector.insert_strided_slice %[[packed2]], %[[out1]] {offsets = [4], strides = [1]} : vector<2xf16> into vector<9xf16>
  // CHECK: %[[elem6:.*]] = vector.extract %[[value]]
  // CHECK: %[[elem7:.*]] = vector.extract %[[value]]
  // CHECK: %[[packed3:.*]] = rocdl.cvt.pkrtz %[[elem6]], %[[elem7]] : vector<2xf16>
  // CHECK: %[[out3:.*]] = vector.insert_strided_slice %[[packed3]], %[[out2]] {offsets = [6], strides = [1]} : vector<2xf16> into vector<9xf16>
  // CHECK: %[[elem8:.*]] = vector.extract %[[value]]
  // CHECK: %[[packed4:.*]] = rocdl.cvt.pkrtz %[[elem8:.*]] : vector<2xf16>
  // CHECK: %[[slice:.*]] = vector.extract_strided_slice %[[packed4]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
  // CHECK: %[[out4:.*]] = vector.insert_strided_slice %[[slice]], %[[out3]] {offsets = [8], strides = [1]} : vector<1xf16> into vector<9xf16>
  // CHECK: return %[[out4]]
  %w = arith.truncf %v : vector<9xf32> to vector<9xf16>
  return %w : vector<9xf16>
}
