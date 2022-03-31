//===- Trait.h - Trait wrapper class ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Trait wrapper to simplify using TableGen Record defining an MLIR Trait.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_TRAIT_H_
#define MLIR_TABLEGEN_TRAIT_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace llvm {
class Init;
class Record;
} // namespace llvm

namespace mlir {
namespace tblgen {

class Interface;

// Wrapper class with helper methods for accessing Trait constraints defined in
// TableGen.
class Trait {
public:
  // Discriminator for kinds of traits.
  enum class Kind {
    // Trait corresponding to C++ class.
    Native,
    // Trait corresponding to a predicate.
    Pred,
    // Trait controlling definition generator internals.
    Internal,
    // Trait corresponding to an Interface.
    Interface
  };

  explicit Trait(Kind kind, const llvm::Record *def);

  // Returns an Trait corresponding to the init provided.
  static Trait create(const llvm::Init *init);

  Kind getKind() const { return kind; }

  // Returns the Tablegen definition this operator was constructed from.
  const llvm::Record &getDef() const { return *def; }

protected:
  // The TableGen definition of this trait.
  const llvm::Record *def;
  Kind kind;
};

// Trait corresponding to a native C++ Trait.
class NativeTrait : public Trait {
public:
  // Returns the trait corresponding to a C++ trait class.
  std::string getFullyQualifiedTraitName() const;

  // Returns if this is a structural op trait.
  bool isStructuralOpTrait() const;

  static bool classof(const Trait *t) { return t->getKind() == Kind::Native; }
};

// Trait corresponding to a predicate on the operation.
class PredTrait : public Trait {
public:
  // Returns the template for constructing the predicate.
  std::string getPredTemplate() const;

  // Returns the description of what the predicate is verifying.
  StringRef getSummary() const;

  static bool classof(const Trait *t) { return t->getKind() == Kind::Pred; }
};

// Trait controlling op definition generator internals.
class InternalTrait : public Trait {
public:
  // Returns the trait controlling op definition generator internals.
  StringRef getFullyQualifiedTraitName() const;

  static bool classof(const Trait *t) { return t->getKind() == Kind::Internal; }
};

// Trait corresponding to an OpInterface on the operation.
class InterfaceTrait : public Trait {
public:
  // Returns interface corresponding to the trait.
  Interface getInterface() const;

  // Returns the trait corresponding to a C++ trait class.
  std::string getFullyQualifiedTraitName() const;

  static bool classof(const Trait *t) {
    return t->getKind() == Kind::Interface;
  }

  // Whether the declaration of methods for this trait should be emitted.
  bool shouldDeclareMethods() const;

  // Returns the methods that should always be declared if this interface is
  // emitting declarations.
  std::vector<StringRef> getAlwaysDeclaredMethods() const;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_TRAIT_H_
