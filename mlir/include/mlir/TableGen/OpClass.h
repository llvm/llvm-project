//===- OpClass.h - Helper classes for Op C++ code emission ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines several classes for Op C++ code emission. They are only
// expected to be used by MLIR TableGen backends.
//
// We emit the op declaration and definition into separate files: *Ops.h.inc
// and *Ops.cpp.inc. The former is to be included in the dialect *Ops.h and
// the latter for dialect *Ops.cpp. This way provides a cleaner interface.
//
// In order to do this split, we need to track method signature and
// implementation logic separately. Signature information is used for both
// declaration and definition, while implementation logic is only for
// definition. So we have the following classes for C++ code emission.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_OPCLASS_H_
#define MLIR_TABLEGEN_OPCLASS_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <string>

namespace mlir {
namespace tblgen {
class FmtObjectBase;

// Class for holding the signature of an op's method for C++ code emission
class OpMethodSignature {
public:
  OpMethodSignature(StringRef retType, StringRef name, StringRef params);

  // Writes the signature as a method declaration to the given `os`.
  void writeDeclTo(raw_ostream &os) const;
  // Writes the signature as the start of a method definition to the given `os`.
  // `namePrefix` is the prefix to be prepended to the method name (typically
  // namespaces for qualifying the method definition).
  void writeDefTo(raw_ostream &os, StringRef namePrefix) const;

private:
  // Returns true if the given C++ `type` ends with '&' or '*', or is empty.
  static bool elideSpaceAfterType(StringRef type);

  std::string returnType;
  std::string methodName;
  std::string parameters;
};

// Class for holding the body of an op's method for C++ code emission
class OpMethodBody {
public:
  explicit OpMethodBody(bool declOnly);

  OpMethodBody &operator<<(Twine content);
  OpMethodBody &operator<<(int content);
  OpMethodBody &operator<<(const FmtObjectBase &content);

  void writeTo(raw_ostream &os) const;

private:
  // Whether this class should record method body.
  bool isEffective;
  std::string body;
};

// Class for holding an op's method for C++ code emission
class OpMethod {
public:
  // Properties (qualifiers) of class methods. Bitfield is used here to help
  // querying properties.
  enum Property {
    MP_None = 0x0,
    MP_Static = 0x1,      // Static method
    MP_Constructor = 0x2, // Constructor
    MP_Private = 0x4,     // Private method
  };

  OpMethod(StringRef retType, StringRef name, StringRef params,
           Property property, bool declOnly);

  OpMethodBody &body();

  // Returns true if this is a static method.
  bool isStatic() const;

  // Returns true if this is a private method.
  bool isPrivate() const;

  // Writes the method as a declaration to the given `os`.
  void writeDeclTo(raw_ostream &os) const;
  // Writes the method as a definition to the given `os`. `namePrefix` is the
  // prefix to be prepended to the method name (typically namespaces for
  // qualifying the method definition).
  void writeDefTo(raw_ostream &os, StringRef namePrefix) const;

private:
  Property properties;
  // Whether this method only contains a declaration.
  bool isDeclOnly;
  OpMethodSignature methodSignature;
  OpMethodBody methodBody;
};

// A class used to emit C++ classes from Tablegen.  Contains a list of public
// methods and a list of private fields to be emitted.
class Class {
public:
  explicit Class(StringRef name);

  // Creates a new method in this class.
  OpMethod &newMethod(StringRef retType, StringRef name, StringRef params = "",
                      OpMethod::Property = OpMethod::MP_None,
                      bool declOnly = false);

  OpMethod &newConstructor(StringRef params = "", bool declOnly = false);

  // Creates a new field in this class.
  void newField(StringRef type, StringRef name, StringRef defaultValue = "");

  // Writes this op's class as a declaration to the given `os`.
  void writeDeclTo(raw_ostream &os) const;
  // Writes the method definitions in this op's class to the given `os`.
  void writeDefTo(raw_ostream &os) const;

  // Returns the C++ class name of the op.
  StringRef getClassName() const { return className; }

protected:
  std::string className;
  SmallVector<OpMethod, 8> methods;
  SmallVector<std::string, 4> fields;
};

// Class for holding an op for C++ code emission
class OpClass : public Class {
public:
  explicit OpClass(StringRef name, StringRef extraClassDeclaration = "");

  // Adds an op trait.
  void addTrait(Twine trait);

  // Writes this op's class as a declaration to the given `os`.  Redefines
  // Class::writeDeclTo to also emit traits and extra class declarations.
  void writeDeclTo(raw_ostream &os) const;

private:
  StringRef extraClassDeclaration;
  SmallVector<std::string, 4> traitsVec;
  StringSet<> traitsSet;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_OPCLASS_H_
