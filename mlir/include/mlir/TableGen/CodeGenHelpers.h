//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for generating C++ from tablegen
// structures.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_CODEGENHELPERS_H
#define MLIR_TABLEGEN_CODEGENHELPERS_H

#include "mlir/TableGen/Dialect.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace tblgen {

// Simple RAII helper for defining ifdef-undef-endif scopes.
class IfDefScope {
public:
  IfDefScope(llvm::StringRef name, llvm::raw_ostream &os)
      : name(name.str()), os(os) {
    os << "#ifdef " << name << "\n"
       << "#undef " << name << "\n\n";
  }
  ~IfDefScope() { os << "\n#endif  // " << name << "\n\n"; }

private:
  std::string name;
  llvm::raw_ostream &os;
};

// A helper RAII class to emit nested namespaces for this op.
class NamespaceEmitter {
public:
  NamespaceEmitter(raw_ostream &os, const Dialect &dialect) : os(os) {
    if (!dialect)
      return;
    emitNamespaceStarts(os, dialect.getCppNamespace());
  }
  NamespaceEmitter(raw_ostream &os, StringRef cppNamespace) : os(os) {
    emitNamespaceStarts(os, cppNamespace);
  }

  ~NamespaceEmitter() {
    for (StringRef ns : llvm::reverse(namespaces))
      os << "} // namespace " << ns << "\n";
  }

private:
  void emitNamespaceStarts(raw_ostream &os, StringRef cppNamespace) {
    llvm::SplitString(cppNamespace, namespaces, "::");
    for (StringRef ns : namespaces)
      os << "namespace " << ns << " {\n";
  }
  raw_ostream &os;
  SmallVector<StringRef, 2> namespaces;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_CODEGENHELPERS_H
