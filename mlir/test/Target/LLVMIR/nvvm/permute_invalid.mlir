// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

llvm.func @invalid_default_missing_hi(%lo: i32, %sel: i32) -> i32 {
  // expected-error @below {{mode 'default' requires 'hi' operand i.e. it requires 3 operands - lo, hi, selector}}
  %r = nvvm.prmt #nvvm.permute_mode<default> %lo, %sel : i32
  llvm.return %r : i32
}

llvm.func @invalid_f4e_missing_hi(%lo: i32, %sel: i32) -> i32 {
  // expected-error @below {{mode 'f4e' requires 'hi' operand i.e. it requires 3 operands - lo, hi, selector}}
  %r = nvvm.prmt #nvvm.permute_mode<f4e> %lo, %sel : i32
  llvm.return %r : i32
}

llvm.func @invalid_b4e_missing_hi(%lo: i32, %sel: i32) -> i32 {
  // expected-error @below {{mode 'b4e' requires 'hi' operand i.e. it requires 3 operands - lo, hi, selector}}
  %r = nvvm.prmt #nvvm.permute_mode<b4e> %lo, %sel : i32
  llvm.return %r : i32
}

llvm.func @invalid_rc8_with_hi(%lo: i32, %sel: i32, %hi: i32) -> i32 {
  // expected-error @below {{mode 'rc8' does not accept 'hi' operand i.e. it requires 2 operands - lo, selector}}
  %r = nvvm.prmt #nvvm.permute_mode<rc8> %lo, %sel, %hi : i32
  llvm.return %r : i32
}

llvm.func @invalid_ecl_with_hi(%lo: i32, %sel: i32, %hi: i32) -> i32 {
  // expected-error @below {{mode 'ecl' does not accept 'hi' operand i.e. it requires 2 operands - lo, selector}}
  %r = nvvm.prmt #nvvm.permute_mode<ecl> %lo, %sel, %hi : i32
  llvm.return %r : i32
}

llvm.func @invalid_ecr_with_hi(%lo: i32, %sel: i32, %hi: i32) -> i32 {
  // expected-error @below {{mode 'ecr' does not accept 'hi' operand i.e. it requires 2 operands - lo, selector}}
  %r = nvvm.prmt #nvvm.permute_mode<ecr> %lo, %sel, %hi : i32
  llvm.return %r : i32
}

llvm.func @invalid_rc16_with_hi(%lo: i32, %sel: i32, %hi: i32) -> i32 {
  // expected-error @below {{mode 'rc16' does not accept 'hi' operand i.e. it requires 2 operands - lo, selector}}
  %r = nvvm.prmt #nvvm.permute_mode<rc16> %lo, %sel, %hi : i32
  llvm.return %r : i32
}
