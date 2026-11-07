// RUN: mlir-opt %s --math-expand-ops=ops=round --convert-vector-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --convert-ub-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 | FileCheck %s
// RUN: mlir-opt %s --math-expand-ops=ops=round --convert-vector-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --convert-ub-to-llvm --reconcile-unrealized-casts | mlir-runner -O2 -e main -entry-point-result=i32 | FileCheck %s

// CHECK: {{^0$}}
// Keep round's operand dynamic until after expansion.
func.func private @round_bits(%x: vector<8xf32>) -> vector<8xi32> {
  %r = math.round %x : vector<8xf32>
  %bits = arith.bitcast %r : vector<8xf32> to vector<8xi32>
  return %bits : vector<8xi32>
}

func.func @main() -> i32 {
  // Half predecessors, exact halves, signed zeros and signed subnormals.
  %input = arith.constant dense<[0x3EFFFFFF, 0xBEFFFFFF, 0x3F000000, 0xBF000000, 0x00000000, 0x80000000, 0x00000001, 0x80000001]> : vector<8xi32>
  %expected = arith.constant dense<[0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x00000000, 0x80000000, 0x00000000, 0x80000000]> : vector<8xi32>
  %x = arith.bitcast %input : vector<8xi32> to vector<8xf32>
  %actual = func.call @round_bits(%x) : (vector<8xf32>) -> vector<8xi32>
  %bad = arith.cmpi ne, %actual, %expected : vector<8xi32>
  %bad_i32 = arith.extui %bad : vector<8xi1> to vector<8xi32>
  %count = vector.reduction <add>, %bad_i32 : vector<8xi32> into i32
  return %count : i32
}
