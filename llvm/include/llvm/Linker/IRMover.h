//===- IRMover.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LINKER_IRMOVER_H
#define LLVM_LINKER_IRMOVER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Compiler.h"
#include <functional>

namespace llvm {
class Error;
class GlobalValue;
class Metadata;
class MDNode;
class NamedMDNode;
class Module;
class StructType;
class TrackingMDRef;
class Type;

class IRMover {
  struct StructTypeKeyInfo {
    struct KeyTy {
      ArrayRef<Type *> ETypes;
      bool IsPacked;
      LLVM_ABI KeyTy(ArrayRef<Type *> E, bool P);
      LLVM_ABI KeyTy(const StructType *ST);
      LLVM_ABI bool operator==(const KeyTy &that) const;
      LLVM_ABI bool operator!=(const KeyTy &that) const;
    };
    LLVM_ABI static StructType *getEmptyKey();
    LLVM_ABI static StructType *getTombstoneKey();
    LLVM_ABI static unsigned getHashValue(const KeyTy &Key);
    LLVM_ABI static unsigned getHashValue(const StructType *ST);
    LLVM_ABI static bool isEqual(const KeyTy &LHS, const StructType *RHS);
    LLVM_ABI static bool isEqual(const StructType *LHS, const StructType *RHS);
  };

  /// Type of the Metadata map in \a ValueToValueMapTy.
  typedef DenseMap<const Metadata *, TrackingMDRef> MDMapT;

public:
  class IdentifiedStructTypeSet {
    // The set of opaque types is the composite module.
    DenseSet<StructType *> OpaqueStructTypes;

    // The set of identified but non opaque structures in the composite module.
    DenseSet<StructType *, StructTypeKeyInfo> NonOpaqueStructTypes;

  public:
    LLVM_ABI void addNonOpaque(StructType *Ty);
    LLVM_ABI void switchToNonOpaque(StructType *Ty);
    LLVM_ABI void addOpaque(StructType *Ty);
    LLVM_ABI StructType *findNonOpaque(ArrayRef<Type *> ETypes, bool IsPacked);
    LLVM_ABI bool hasType(StructType *Ty);
  };

  LLVM_ABI IRMover(Module &M);

  typedef std::function<void(GlobalValue &)> ValueAdder;
  using LazyCallback =
      llvm::unique_function<void(GlobalValue &GV, ValueAdder Add)>;

  using NamedMDNodesT = DenseMap<const NamedMDNode *, DenseSet<const MDNode *>>;

  /// Move in the provide values in \p ValuesToLink from \p Src.
  ///
  /// - \p AddLazyFor is a call back that the IRMover will call when a global
  ///   value is referenced by one of the ValuesToLink (transitively) but was
  ///   not present in ValuesToLink. The GlobalValue and a ValueAdder callback
  ///   are passed as an argument, and the callback is expected to be called
  ///   if the GlobalValue needs to be added to the \p ValuesToLink and linked.
  ///   Pass nullptr if there's no work to be done in such cases.
  /// - \p IsPerformingImport is true when this IR link is to perform ThinLTO
  ///   function importing from Src.
  LLVM_ABI Error move(std::unique_ptr<Module> Src,
                      ArrayRef<GlobalValue *> ValuesToLink,
                      LazyCallback AddLazyFor, bool IsPerformingImport);
  Module &getModule() { return Composite; }

private:
  Module &Composite;
  IdentifiedStructTypeSet IdentifiedStructTypes;
  MDMapT SharedMDs; ///< A Metadata map to use for all calls to \a move().
  NamedMDNodesT NamedMDNodes; ///< Cache for IRMover::linkNamedMDNodes().
};

} // End llvm namespace

#endif
