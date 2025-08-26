//===- WasmBinaryEncoding.h - Byte encodings for Wasm binary format ===----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Define various flags used to encode instructions, types, etc. in
// WebAssembly binary format.
//
// These encodings are defined in the WebAssembly binary format specification.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_TARGET_WASMBINARYENCODING
#define MLIR_TARGET_WASMBINARYENCODING

#include <cstddef>

namespace mlir {
struct WasmBinaryEncoding {
  /// Byte encodings for Wasm instructions.
  struct OpCode {
    // Locals, globals, constants.
    static constexpr std::byte localGet{0x20};
    static constexpr std::byte localSet{0x21};
    static constexpr std::byte localTee{0x22};
    static constexpr std::byte globalGet{0x23};
    static constexpr std::byte constI32{0x41};
    static constexpr std::byte constI64{0x42};
    static constexpr std::byte constFP32{0x43};
    static constexpr std::byte constFP64{0x44};

    // Numeric operations.
    static constexpr std::byte clzI32{0x67};
    static constexpr std::byte ctzI32{0x68};
    static constexpr std::byte popcntI32{0x69};
    static constexpr std::byte addI32{0x6A};
    static constexpr std::byte subI32{0x6B};
    static constexpr std::byte mulI32{0x6C};
    static constexpr std::byte divSI32{0x6d};
    static constexpr std::byte divUI32{0x6e};
    static constexpr std::byte remSI32{0x6f};
    static constexpr std::byte remUI32{0x70};
    static constexpr std::byte andI32{0x71};
    static constexpr std::byte orI32{0x72};
    static constexpr std::byte xorI32{0x73};
    static constexpr std::byte shlI32{0x74};
    static constexpr std::byte shrSI32{0x75};
    static constexpr std::byte shrUI32{0x76};
    static constexpr std::byte rotlI32{0x77};
    static constexpr std::byte rotrI32{0x78};
    static constexpr std::byte clzI64{0x79};
    static constexpr std::byte ctzI64{0x7A};
    static constexpr std::byte popcntI64{0x7B};
    static constexpr std::byte addI64{0x7C};
    static constexpr std::byte subI64{0x7D};
    static constexpr std::byte mulI64{0x7E};
    static constexpr std::byte divSI64{0x7F};
    static constexpr std::byte divUI64{0x80};
    static constexpr std::byte remSI64{0x81};
    static constexpr std::byte remUI64{0x82};
    static constexpr std::byte andI64{0x83};
    static constexpr std::byte orI64{0x84};
    static constexpr std::byte xorI64{0x85};
    static constexpr std::byte shlI64{0x86};
    static constexpr std::byte shrSI64{0x87};
    static constexpr std::byte shrUI64{0x88};
    static constexpr std::byte rotlI64{0x89};
    static constexpr std::byte rotrI64{0x8A};
    static constexpr std::byte absF32{0x8B};
    static constexpr std::byte negF32{0x8C};
    static constexpr std::byte ceilF32{0x8D};
    static constexpr std::byte floorF32{0x8E};
    static constexpr std::byte truncF32{0x8F};
    static constexpr std::byte sqrtF32{0x91};
    static constexpr std::byte addF32{0x92};
    static constexpr std::byte subF32{0x93};
    static constexpr std::byte mulF32{0x94};
    static constexpr std::byte divF32{0x95};
    static constexpr std::byte minF32{0x96};
    static constexpr std::byte maxF32{0x97};
    static constexpr std::byte copysignF32{0x98};
    static constexpr std::byte absF64{0x99};
    static constexpr std::byte negF64{0x9A};
    static constexpr std::byte ceilF64{0x9B};
    static constexpr std::byte floorF64{0x9C};
    static constexpr std::byte truncF64{0x9D};
    static constexpr std::byte sqrtF64{0x9F};
    static constexpr std::byte addF64{0xA0};
    static constexpr std::byte subF64{0xA1};
    static constexpr std::byte mulF64{0xA2};
    static constexpr std::byte divF64{0xA3};
    static constexpr std::byte minF64{0xA4};
    static constexpr std::byte maxF64{0xA5};
    static constexpr std::byte copysignF64{0xA6};
    static constexpr std::byte wrap{0xA7};
  };

  /// Byte encodings of types in Wasm binaries
  struct Type {
    static constexpr std::byte emptyBlockType{0x40};
    static constexpr std::byte funcType{0x60};
    static constexpr std::byte externRef{0x6F};
    static constexpr std::byte funcRef{0x70};
    static constexpr std::byte v128{0x7B};
    static constexpr std::byte f64{0x7C};
    static constexpr std::byte f32{0x7D};
    static constexpr std::byte i64{0x7E};
    static constexpr std::byte i32{0x7F};
  };

  /// Byte encodings of Wasm imports.
  struct Import {
    static constexpr std::byte typeID{0x00};
    static constexpr std::byte tableType{0x01};
    static constexpr std::byte memType{0x02};
    static constexpr std::byte globalType{0x03};
  };

  /// Byte encodings for Wasm limits.
  struct LimitHeader {
    static constexpr std::byte lowLimitOnly{0x00};
    static constexpr std::byte bothLimits{0x01};
  };

  /// Byte encodings describing the mutability of globals.
  struct GlobalMutability {
    static constexpr std::byte isConst{0x00};
    static constexpr std::byte isMutable{0x01};
  };

  /// Byte encodings describing Wasm exports.
  struct Export {
    static constexpr std::byte function{0x00};
    static constexpr std::byte table{0x01};
    static constexpr std::byte memory{0x02};
    static constexpr std::byte global{0x03};
  };

  static constexpr std::byte endByte{0x0B};
};
} // namespace mlir

#endif
