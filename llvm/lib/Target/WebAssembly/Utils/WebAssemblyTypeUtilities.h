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

#include "MCTargetDesc/WebAssemblyMCTypeUtilities.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/Support/MachineValueType.h"

namespace llvm {

class TargetRegisterClass;

namespace WebAssembly {

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

// Convert StringRef to ValType / HealType / BlockType

MVT parseMVT(StringRef Type);

// Convert a MVT into its corresponding wasm ValType.
wasm::ValType toValType(MVT Type);

// Convert a register class to a wasm ValType.
wasm::ValType regClassToValType(const TargetRegisterClass *RC);

/// Sets a Wasm Symbol Type.
void wasmSymbolSetType(MCSymbolWasm *Sym, const Type *GlobalVT,
                       const ArrayRef<MVT> &VTs);

} // end namespace WebAssembly
} // end namespace llvm

#endif
