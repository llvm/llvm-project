//===- LLVMAttrs.cpp - LLVM Attributes registration -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the attribute details for the LLVM IR dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include <optional>

using namespace mlir;
using namespace mlir::LLVM;

#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// LLVMDialect registration
//===----------------------------------------------------------------------===//

void LLVMDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DINodeAttr
//===----------------------------------------------------------------------===//

bool DINodeAttr::classof(Attribute attr) {
  return llvm::isa<DIBasicTypeAttr, DICompileUnitAttr, DICompositeTypeAttr,
                   DIDerivedTypeAttr, DIFileAttr, DILabelAttr,
                   DILexicalBlockAttr, DILexicalBlockFileAttr,
                   DILocalVariableAttr, DINamespaceAttr, DINullTypeAttr,
                   DISubprogramAttr, DISubrangeAttr, DISubroutineTypeAttr>(
      attr);
}

//===----------------------------------------------------------------------===//
// DIScopeAttr
//===----------------------------------------------------------------------===//

bool DIScopeAttr::classof(Attribute attr) {
  return llvm::isa<DICompileUnitAttr, DICompositeTypeAttr, DIFileAttr,
                   DILocalScopeAttr, DINamespaceAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DILocalScopeAttr
//===----------------------------------------------------------------------===//

bool DILocalScopeAttr::classof(Attribute attr) {
  return llvm::isa<DILexicalBlockAttr, DILexicalBlockFileAttr,
                   DISubprogramAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DITypeAttr
//===----------------------------------------------------------------------===//

bool DITypeAttr::classof(Attribute attr) {
  return llvm::isa<DINullTypeAttr, DIBasicTypeAttr, DICompositeTypeAttr,
                   DIDerivedTypeAttr, DISubroutineTypeAttr>(attr);
}

//===----------------------------------------------------------------------===//
// TBAANodeAttr
//===----------------------------------------------------------------------===//

bool TBAANodeAttr::classof(Attribute attr) {
  return llvm::isa<TBAATypeDescriptorAttr, TBAARootAttr>(attr);
}

//===----------------------------------------------------------------------===//
// MemoryEffectsAttr
//===----------------------------------------------------------------------===//

MemoryEffectsAttr MemoryEffectsAttr::get(MLIRContext *context,
                                         ArrayRef<ModRefInfo> memInfoArgs) {
  if (memInfoArgs.empty())
    return MemoryEffectsAttr::get(context, ModRefInfo::ModRef,
                                  ModRefInfo::ModRef, ModRefInfo::ModRef);
  if (memInfoArgs.size() == 3)
    return MemoryEffectsAttr::get(context, memInfoArgs[0], memInfoArgs[1],
                                  memInfoArgs[2]);
  return {};
}

bool MemoryEffectsAttr::isReadWrite() {
  if (this->getArgMem() != ModRefInfo::ModRef)
    return false;
  if (this->getInaccessibleMem() != ModRefInfo::ModRef)
    return false;
  if (this->getOther() != ModRefInfo::ModRef)
    return false;
  return true;
}
