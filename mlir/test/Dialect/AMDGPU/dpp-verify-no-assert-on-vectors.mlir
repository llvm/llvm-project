// RUN: mlir-opt %s -verify-each

// DPPOp verifier must not assert when src type is a
// vector (e.g. ARM SME tile vectors).

module {
  func.func @main() {
    %tile = arm_sme.get_tile : vector<[16]x[16]xi8>
    %pop = math.ctpop %tile : vector<[16]x[16]xi8>
    %r = amdgpu.dpp %pop %tile row_shl(1 : i32) : vector<[16]x[16]xi8>
    return
  }
}
