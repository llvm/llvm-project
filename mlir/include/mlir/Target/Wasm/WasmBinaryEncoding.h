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
    static constexpr std::byte constI32{0x41};
    static constexpr std::byte constI64{0x42};
    static constexpr std::byte constFP32{0x43};
    static constexpr std::byte constFP64{0x44};
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
