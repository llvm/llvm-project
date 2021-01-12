//===- Builder.h - Builder classes ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder wrapper to simplify using TableGen Record for building
// operations/types/etc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_BUILDER_H_
#define MLIR_TABLEGEN_BUILDER_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Init;
class Record;
class SMLoc;
} // end namespace llvm

namespace mlir {
namespace tblgen {

/// Wrapper class with helper methods for accessing Builders defined in
/// TableGen.
class Builder {
public:
  /// This class represents a single parameter to a builder method.
  class Parameter {
  public:
    /// Return a string containing the C++ type of this parameter.
    StringRef getCppType() const;

    /// Return an optional string containing the name of this parameter. If
    /// None, no name was specified for this parameter by the user.
    Optional<StringRef> getName() const { return name; }

    /// Return an optional string containing the default value to use for this
    /// parameter.
    Optional<StringRef> getDefaultValue() const;

  private:
    Parameter(Optional<StringRef> name, const llvm::Init *def)
        : name(name), def(def) {}

    /// The optional name of the parameter.
    Optional<StringRef> name;

    /// The tablegen definition of the parameter. This is either a StringInit,
    /// or a CArg DefInit.
    const llvm::Init *def;

    // Allow access to the constructor.
    friend Builder;
  };

  /// Construct a builder from the given Record instance.
  Builder(const llvm::Record *record, ArrayRef<llvm::SMLoc> loc);

  /// Return a list of parameters used in this build method.
  ArrayRef<Parameter> getParameters() const { return parameters; }

  /// Return an optional string containing the body of the builder.
  Optional<StringRef> getBody() const;

protected:
  /// The TableGen definition of this builder.
  const llvm::Record *def;

private:
  /// A collection of parameters to the builder.
  SmallVector<Parameter> parameters;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_BUILDER_H_
