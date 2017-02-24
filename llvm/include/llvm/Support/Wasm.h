//===- Wasm.h - Wasm object file format -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines manifest constants for the wasm object file format.
// See: https://github.com/WebAssembly/design/blob/master/BinaryEncoding.md
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_WASM_H
#define LLVM_SUPPORT_WASM_H

#include "llvm/ADT/ArrayRef.h"

namespace llvm {
namespace wasm {

// Object file magic string.
const char WasmMagic[] = {'\0', 'a', 's', 'm'};
// Wasm binary format version
const uint32_t WasmVersion = 0x1;

struct WasmObjectHeader {
  StringRef Magic;
  uint32_t Version;
};

struct WasmSection {
  uint32_t Type;             // Section type (See below)
  uint32_t Offset;           // Offset with in the file
  StringRef Name;            // Section name (User-defined sections only)
  ArrayRef<uint8_t> Content; // Section content
};

enum : unsigned {
  WASM_SEC_CUSTOM = 0,   // Custom / User-defined section
  WASM_SEC_TYPE = 1,     // Function signature declarations
  WASM_SEC_IMPORT = 2,   // Import declarations
  WASM_SEC_FUNCTION = 3, // Function declarations
  WASM_SEC_TABLE = 4,    // Indirect function table and other tables
  WASM_SEC_MEMORY = 5,   // Memory attributes
  WASM_SEC_GLOBAL = 6,   // Global declarations
  WASM_SEC_EXPORT = 7,   // Exports
  WASM_SEC_START = 8,    // Start function declaration
  WASM_SEC_ELEM = 9,     // Elements section
  WASM_SEC_CODE = 10,    // Function bodies (code)
  WASM_SEC_DATA = 11     // Data segments
};

// Type immediate encodings used in various contexts.
enum : unsigned {
  WASM_TYPE_I32          = 0x7f,
  WASM_TYPE_I64          = 0x7e,
  WASM_TYPE_F32          = 0x7d,
  WASM_TYPE_F64          = 0x7c,
  WASM_TYPE_ANYFUNC      = 0x70,
  WASM_TYPE_FUNC         = 0x60,
  WASM_TYPE_NORESULT     = 0x40, // for blocks with no result values
};

// Kinds of externals (for imports and exports).
enum : unsigned {
  WASM_EXTERNAL_FUNCTION = 0x0,
  WASM_EXTERNAL_TABLE    = 0x1,
  WASM_EXTERNAL_MEMORY   = 0x2,
  WASM_EXTERNAL_GLOBAL   = 0x3,
};

// Opcodes used in initializer expressions.
enum : unsigned {
  WASM_OPCODE_END        = 0x0b,
  WASM_OPCODE_GET_GLOBAL = 0x23,
  WASM_OPCODE_I32_CONST  = 0x41,
  WASM_OPCODE_I64_CONST  = 0x42,
  WASM_OPCODE_F32_CONST  = 0x43,
  WASM_OPCODE_F64_CONST  = 0x44,
};

#define WASM_RELOC(name, value) name = value,

enum : unsigned {
#include "WasmRelocs/WebAssembly.def"
};

} // end namespace wasm
} // end namespace llvm

#endif
