//===-- DebugLoc.cpp - Implement DebugLoc class ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugLoc.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include <optional>
using namespace llvm;

//===----------------------------------------------------------------------===//
// DebugLoc Implementation
//===----------------------------------------------------------------------===//

Metadata *DebugLoc::toMetadata(LLVMContext &C) const {
  return ConstantAsMetadata::get(ConstantInt::get(llvm::Type::getInt64Ty(C), getAsRawInteger()));
}
std::optional<DebugLoc> DebugLoc::fromMetadata(Metadata *MD) {
  if (auto *CAM = dyn_cast<ConstantAsMetadata>(MD))
    if (auto *CI = dyn_cast<ConstantInt>(CAM->getValue()))
      return DebugLoc(CI->getZExtValue());
  return std::nullopt;
}

MDNode *DebugLoc::getTransientDILocation(DISubprogram *SP) {
  if (!isUsingTransientDILocation())
    return nullptr;
  return nullptr;
  // return SP->TransientDILocations[SrcLocIndex];
}

unsigned DebugLoc::getLine(DISubprogram *SP) const {
  assert(SP && *this && "Expected valid DebugLoc");
  return SP->getSrcLoc(*this).Line;
}

unsigned DebugLoc::getCol(DISubprogram *SP) const {
  assert(SP && *this && "Expected valid DebugLoc");
  return SP->getSrcLoc(*this).Column;
}

DILocalScope *DebugLoc::getScope(DISubprogram *SP) const {
  assert(SP && *this && "Expected valid DebugLoc");
  return SP->getLocScope(*this).Scope;
}

DebugLoc DebugLoc::getInlinedAt(DISubprogram *SP) const {
  assert(SP && *this && "Expected valid DebugLoc");
  DebugLoc InlinedAt = SP->getLocScope(*this).InlinedAt;
  return DebugLoc(InlinedAt.SrcLocIndex, InlinedAt.LocScopeIndex);
}

uint64_t DebugLoc::getAtomGroup(DISubprogram *SP) const {
  return SP->getSrcLoc(*this).AtomGroup;
}
uint8_t DebugLoc::getAtomRank(DISubprogram *SP) const {
  return SP->getSrcLoc(*this).AtomRank;
}

DILocalScope *DebugLoc::getInlinedAtScope(DISubprogram *SP) const {
  if (!SP)
    return nullptr;
  return SP->getInlinedAtScope(*this);
}

bool DebugLoc::isImplicitCode(DISubprogram *SP) const {
  if (!SP)
    return true;
  return SP->getSrcLoc(DebugLoc(SrcLocIndex, LocScopeIndex)).IsImplicitCode;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void DebugLoc::dump() const { print(dbgs()); }
#endif

void DebugLoc::print(raw_ostream &OS) const {
  if (!isValid())
    return;

  // Print indices without source info.
  OS << "DebugLoc(Src: " << SrcLocIndex << ", Scope: " << LocScopeIndex << ")";
}

void DebugLoc::print(raw_ostream &OS, DISubprogram *SP) const {
  if (!isValid())
    return;

  // Print source location info using SP.
  auto *Scope = getScope(SP);
  OS << Scope->getFilename();
  OS << ':' << getLine(SP);
  if (getCol(SP) != 0)
    OS << ':' << getCol(SP);

  if (DebugLoc InlinedAtDL = getInlinedAt(SP)) {
    OS << " @[ ";
    InlinedAtDL.print(OS);
    OS << " ]";
  }
}
