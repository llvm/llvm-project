//===- TypeID.cpp - MLIR TypeID -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/TypeID.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/RWMutex.h"

#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "typeid"

//===----------------------------------------------------------------------===//
// TypeID Registry
//===----------------------------------------------------------------------===//

namespace {
struct ImplicitTypeIDRegistry {
  /// Lookup or insert a TypeID for the given type name.
  TypeID lookupOrInsert(StringRef typeName) {
    LLVM_DEBUG(llvm::dbgs() << "ImplicitTypeIDRegistry::lookupOrInsert("
                            << typeName << ")\n");

    // Perform a heuristic check to see if this type is in an anonymous
    // namespace. String equality is not valid for anonymous types, so we try to
    // abort whenever we see them.
#ifndef NDEBUG
#if defined(_MSC_VER)
    if (typeName.contains("anonymous-namespace")) {
#else
    if (typeName.contains("anonymous namespace")) {
#endif
      std::string errorStr;
      {
        llvm::raw_string_ostream errorOS(errorStr);
        errorOS << "TypeID::get<" << typeName
                << ">(): Using TypeID on a class with an anonymous "
                   "namespace requires an explicit TypeID definition. The "
                   "implicit fallback uses string name, which does not "
                   "guarantee uniqueness in anonymous contexts. Define an "
                   "explicit TypeID instantiation for this type using "
                   "`MLIR_DECLARE_EXPLICIT_TYPE_ID`/"
                   "`MLIR_DEFINE_EXPLICIT_TYPE_ID` or "
                   "`MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID`.\n";
      }
      llvm::report_fatal_error(errorStr);
    }
#endif

    { // Try a read-only lookup first.
      llvm::sys::SmartScopedReader<true> guard(mutex);
      auto it = typeNameToID.find(typeName);
      if (it != typeNameToID.end())
        return it->second;
    }
    llvm::sys::SmartScopedWriter<true> guard(mutex);
    auto it = typeNameToID.try_emplace(typeName, TypeID());
    if (it.second)
      it.first->second = typeIDAllocator.allocate();
    return it.first->second;
  }

  /// A mutex that guards access to the registry.
  llvm::sys::SmartRWMutex<true> mutex;

  /// An allocator used for TypeID objects.
  TypeIDAllocator typeIDAllocator;

  /// A map type name to TypeID.
  DenseMap<StringRef, TypeID> typeNameToID;
};
} // end namespace

TypeID detail::FallbackTypeIDResolver::registerImplicitTypeID(StringRef name) {
  // To prevent race conditions when one thread is accessing this `static`
  // variable while other thread destructing it; construct the `registry`
  // on the heap.
  static auto *registry = new ImplicitTypeIDRegistry();
  return registry->lookupOrInsert(name);
}

//===----------------------------------------------------------------------===//
// Builtin TypeIDs
//===----------------------------------------------------------------------===//

MLIR_DEFINE_EXPLICIT_TYPE_ID(void)
