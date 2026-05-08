// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

llvm.func @invalid_default_missing_hi(%sel: i32, %lo: i32) -> i32 {
  // expected-error @below {{mode 'default' requires 'hi' operand.}}
  %r = nvvm.prmt #nvvm.permute_mode<default> %sel, %lo : i32
  llvm.return %r : i32
}

llvm.func @invalid_f4e_missing_hi(%sel: i32, %lo: i32) -> i32 {
  // expected-error @below {{mode 'f4e' requires 'hi' operand.}}
  %r = nvvm.prmt #nvvm.permute_mode<f4e> %sel, %lo : i32
  llvm.return %r : i32
}

llvm.func @invalid_b4e_missing_hi(%sel: i32, %lo: i32) -> i32 {
  // expected-error @below {{mode 'b4e' requires 'hi' operand.}}
  %r = nvvm.prmt #nvvm.permute_mode<b4e> %sel, %lo : i32
  llvm.return %r : i32
}

llvm.func @invalid_rc8_with_hi(%sel: i32, %lo: i32, %hi: i32) -> i32 {
  // expected-error @below {{mode 'rc8' does not accept 'hi' operand.}}
  %r = nvvm.prmt #nvvm.permute_mode<rc8> %sel, %lo, %hi : i32
  llvm.return %r : i32
}

llvm.func @invalid_ecl_with_hi(%sel: i32, %lo: i32, %hi: i32) -> i32 {
  // expected-error @below {{mode 'ecl' does not accept 'hi' operand.}}
  %r = nvvm.prmt #nvvm.permute_mode<ecl> %sel, %lo, %hi : i32
  llvm.return %r : i32
}

llvm.func @invalid_ecr_with_hi(%sel: i32, %lo: i32, %hi: i32) -> i32 {
  // expected-error @below {{mode 'ecr' does not accept 'hi' operand.}}
  %r = nvvm.prmt #nvvm.permute_mode<ecr> %sel, %lo, %hi : i32
  llvm.return %r : i32
}

llvm.func @invalid_rc16_with_hi(%sel: i32, %lo: i32, %hi: i32) -> i32 {
  // expected-error @below {{mode 'rc16' does not accept 'hi' operand.}}
  %r = nvvm.prmt #nvvm.permute_mode<rc16> %sel, %lo, %hi : i32
  llvm.return %r : i32
}
