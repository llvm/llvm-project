//===-- WebAssemblyTypeUtilities - WebAssembly Type Utilities---*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the WebAssembly-specific type parsing
/// utility functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_UTILS_WEBASSEMBLYTYPEUTILITIES_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_UTILS_WEBASSEMBLYTYPEUTILITIES_H

#include "llvm/ADT/Optional.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/Support/MachineValueType.h"

namespace llvm {

class TargetRegisterClass;

namespace WebAssembly {

/// Used as immediate MachineOperands for block signatures
enum class BlockType : unsigned {
  Invalid = 0x00,
  Void = 0x40,
  I32 = unsigned(wasm::ValType::I32),
  I64 = unsigned(wasm::ValType::I64),
  F32 = unsigned(wasm::ValType::F32),
  F64 = unsigned(wasm::ValType::F64),
  V128 = unsigned(wasm::ValType::V128),
  Externref = unsigned(wasm::ValType::EXTERNREF),
  Funcref = unsigned(wasm::ValType::FUNCREF),
  // Multivalue blocks (and other non-void blocks) are only emitted when the
  // blocks will never be exited and are at the ends of functions (see
  // WebAssemblyCFGStackify::fixEndsAtEndOfFunction). They also are never made
  // to pop values off the stack, so the exact multivalue signature can always
  // be inferred from the return type of the parent function in MCInstLower.
  Multivalue = 0xffff,
};

enum WasmAddressSpace : unsigned {
  // Default address space, for pointers to linear memory (stack, heap, data).
  WASM_ADDRESS_SPACE_DEFAULT = 0,
  // A non-integral address space for pointers to named objects outside of
  // linear memory: WebAssembly globals or WebAssembly locals.  Loads and stores
  // to these pointers are lowered to global.get / global.set or local.get /
  // local.set, as appropriate.
  WASM_ADDRESS_SPACE_VAR = 1,
  // A non-integral address space for externref values
  WASM_ADDRESS_SPACE_EXTERNREF = 10,
  // A non-integral address space for funcref values
  WASM_ADDRESS_SPACE_FUNCREF = 20,
};

inline bool isDefaultAddressSpace(unsigned AS) {
  return AS == WASM_ADDRESS_SPACE_DEFAULT;
}
inline bool isWasmVarAddressSpace(unsigned AS) {
  return AS == WASM_ADDRESS_SPACE_VAR;
}
inline bool isValidAddressSpace(unsigned AS) {
  return isDefaultAddressSpace(AS) || isWasmVarAddressSpace(AS);
}
inline bool isFuncrefType(const Type *Ty) {
  return isa<PointerType>(Ty) &&
         Ty->getPointerAddressSpace() ==
             WasmAddressSpace::WASM_ADDRESS_SPACE_FUNCREF;
}
inline bool isExternrefType(const Type *Ty) {
  return isa<PointerType>(Ty) &&
         Ty->getPointerAddressSpace() ==
             WasmAddressSpace::WASM_ADDRESS_SPACE_EXTERNREF;
}
inline bool isRefType(const Type *Ty) {
  return isFuncrefType(Ty) || isExternrefType(Ty);
}

inline bool isRefType(wasm::ValType Type) {
  return Type == wasm::ValType::EXTERNREF || Type == wasm::ValType::FUNCREF;
}

// Convert StringRef to ValType / HealType / BlockType

std::optional<wasm::ValType> parseType(StringRef Type);
BlockType parseBlockType(StringRef Type);
MVT parseMVT(StringRef Type);

// Convert ValType or a list/signature of ValTypes to a string.

// Convert an unsinged integer, which can be among wasm::ValType enum, to its
// type name string. If the input is not within wasm::ValType, returns
// "invalid_type".
const char *anyTypeToString(unsigned Type);
const char *typeToString(wasm::ValType Type);
// Convert a list of ValTypes into a string in the format of
// "type0, type1, ... typeN"
std::string typeListToString(ArrayRef<wasm::ValType> List);
// Convert a wasm signature into a string in the format of
// "(params) -> (results)", where params and results are a string of ValType
// lists.
std::string signatureToString(const wasm::WasmSignature *Sig);

// Convert a MVT into its corresponding wasm ValType.
wasm::ValType toValType(MVT Type);

// Convert a register class ID to a wasm ValType.
wasm::ValType regClassToValType(unsigned RC);

// Convert a register class to a wasm ValType.
wasm::ValType regClassToValType(const TargetRegisterClass *RC);

/// Sets a Wasm Symbol Type.
void wasmSymbolSetType(MCSymbolWasm *Sym, const Type *GlobalVT,
                       const SmallVector<MVT, 1> &VTs);

} // end namespace WebAssembly
} // end namespace llvm

#endif
