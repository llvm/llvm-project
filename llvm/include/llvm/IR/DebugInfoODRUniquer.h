//===- llvm/IR/DebugInfoODRUniquer.h - Debug info metadata ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a class used to merge debug info for ODR types.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Metadata.h"

namespace llvm {
class MDString;
class Metadata;
class DICompositeType;

/// Dense set/map find_as key for use alongside DISubprogramODRInfo to
/// merge function declarations of ODR types.
struct DISubprogramODRKey {
  StringRef ScopeIdentifier;
  StringRef LinkageName;

  DISubprogramODRKey(StringRef ScopeIdentifier, StringRef LinkageName)
      : ScopeIdentifier(ScopeIdentifier), LinkageName(LinkageName) {}

  DISubprogramODRKey(const DISubprogram *SP) {
    auto *CT = dyn_cast_or_null<DICompositeType>(SP->getRawScope());
    ScopeIdentifier = CT ? CT->getIdentifier() : StringRef();
    LinkageName = SP->getLinkageName();
  }

  static bool isEqual(const DISubprogramODRKey &LHS, const DISubprogram *RHS) {
    if (LHS.LinkageName.empty() || LHS.ScopeIdentifier.empty())
      return false;

    auto *CT = dyn_cast_or_null<DICompositeType>(RHS->getRawScope());
    if (!CT || !CT->getRawIdentifier())
      return false;

    if (!RHS->getRawLinkageName())
      return false;

    return LHS.LinkageName == RHS->getLinkageName() &&
           LHS.ScopeIdentifier == CT->getIdentifier();
  }

  static bool isEqual(const DISubprogram *LHS, const DISubprogram *RHS) {
    assert(!LHS->isDefinition() && !RHS->isDefinition());
    return isEqual(DISubprogramODRKey(LHS), RHS);
  }
};

/// Dense set/map info to merge function declarations of ODR types.
struct DISubprogramODRInfo {
  static unsigned getHashValue(const DISubprogramODRKey &SP) {
    return hash_combine(SP.ScopeIdentifier, SP.LinkageName);
  }

  static bool isEqual(const DISubprogramODRKey &LHS, const DISubprogram *RHS) {
    return DISubprogramODRKey::isEqual(LHS, RHS);
  }

  static bool isEqual(const DISubprogram *LHS, const DISubprogram *RHS) {
    assert(!LHS->isDefinition() && !RHS->isDefinition());
    return DISubprogramODRKey::isEqual(DISubprogramODRKey(LHS), RHS);
  }
};

class DebugInfoODRUniquer {
  /// Function declarations keyed by DISubprogramODRKey to unique on a subset
  /// of fields, rather than the built-in metadata uniquing which requires
  /// full structural equality.
  DenseSet<DISubprogram *, DISubprogramODRInfo> FnDecls;

public:
  // FIXME: Improve the interface for types.
  DenseMap<const MDString *, DICompositeType *> DITypeMap;

  /// Get an existing DISubprogram declaration with matching scope, linkage
  /// name, type, and template parameters, that has been registered with
  /// `addSubprogramDecl`, or return nullptr.
  LLVM_ABI DISubprogram *getODRSubprogramDecl(Metadata *Scope,
                                              StringRef LinkageName,
                                              Metadata *Type,
                                              Metadata *TemplateParams);

  /// Register function declaration DISubprogram, which may be reused in place
  /// of other ODR-similar DISubprograms (using `getODRSubprogramDecl`).
  LLVM_ABI void addSubprogramDecl(DISubprogram *SP);
};

} // namespace llvm
