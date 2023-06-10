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
#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/CodeGen/WasmAddressSpaces.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/MC/MCSymbolWasm.h"

namespace llvm {

class TargetRegisterClass;

namespace WebAssembly {

/// Return true if this is a WebAssembly Externref Type.
inline bool isWebAssemblyExternrefType(const Type *Ty) {
  return Ty->getPointerAddressSpace() ==
         WebAssembly::WasmAddressSpace::WASM_ADDRESS_SPACE_EXTERNREF;
}

/// Return true if this is a WebAssembly Funcref Type.
inline bool isWebAssemblyFuncrefType(const Type *Ty) {
  return Ty->getPointerAddressSpace() ==
         WebAssembly::WasmAddressSpace::WASM_ADDRESS_SPACE_FUNCREF;
}

/// Return true if this is a WebAssembly Reference Type.
inline bool isWebAssemblyReferenceType(const Type *Ty) {
  return isWebAssemblyExternrefType(Ty) || isWebAssemblyFuncrefType(Ty);
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
