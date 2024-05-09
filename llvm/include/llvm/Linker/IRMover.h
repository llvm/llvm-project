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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/TargetParser/Triple.h"
#include <functional>

namespace llvm {
class Error;
class Metadata;
class Module;
class StructType;
class TrackingMDRef;
class Type;

class IRMover {
  struct StructTypeKeyInfo {
    struct KeyTy {
      ArrayRef<Type *> ETypes;
      bool IsPacked;
      KeyTy(ArrayRef<Type *> E, bool P);
      KeyTy(const StructType *ST);
      bool operator==(const KeyTy &that) const;
      bool operator!=(const KeyTy &that) const;
    };
    static StructType *getEmptyKey();
    static StructType *getTombstoneKey();
    static unsigned getHashValue(const KeyTy &Key);
    static unsigned getHashValue(const StructType *ST);
    static bool isEqual(const KeyTy &LHS, const StructType *RHS);
    static bool isEqual(const StructType *LHS, const StructType *RHS);
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
    void addNonOpaque(StructType *Ty);
    void switchToNonOpaque(StructType *Ty);
    void addOpaque(StructType *Ty);
    StructType *findNonOpaque(ArrayRef<Type *> ETypes, bool IsPacked);
    bool hasType(StructType *Ty);
  };

  /// Utility for handling linking of known libcall functions. If a merged
  /// module contains a recognized library call we can no longer perform any
  /// libcall related transformations.
  class LibcallHandler {
    bool HasLibcalls = false;

    StringSet<> Libcalls;
    StringSet<> Triples;

    BumpPtrAllocator Alloc;
    StringSaver Saver;

  public:
    LibcallHandler() : Saver(Alloc) {}

    void updateLibcalls(const Triple &TheTriple);

    bool checkLibcalls(GlobalValue &GV) {
      if (HasLibcalls)
        return false;
      return HasLibcalls = isa<Function>(&GV) && !GV.isDeclaration() &&
                           Libcalls.count(GV.getName());
    }

    bool hasLibcalls() const { return HasLibcalls; }
  };

  IRMover(Module &M);

  typedef std::function<void(GlobalValue &)> ValueAdder;
  using LazyCallback =
      llvm::unique_function<void(GlobalValue &GV, ValueAdder Add)>;

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
  Error move(std::unique_ptr<Module> Src, ArrayRef<GlobalValue *> ValuesToLink,
             LazyCallback AddLazyFor, bool IsPerformingImport);
  Module &getModule() { return Composite; }

private:
  Module &Composite;
  IdentifiedStructTypeSet IdentifiedStructTypes;
  MDMapT SharedMDs; ///< A Metadata map to use for all calls to \a move().
  LibcallHandler Libcalls;
};

} // End llvm namespace

#endif
