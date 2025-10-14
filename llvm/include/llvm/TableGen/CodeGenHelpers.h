//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for generating C++ code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_CODEGENHELPERS_H
#define LLVM_TABLEGEN_CODEGENHELPERS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace llvm {
// Simple RAII helper for emitting ifdef-undef-endif scope.
class IfDefEmitter {
public:
  IfDefEmitter(raw_ostream &OS, StringRef Name) : Name(Name.str()), OS(OS) {
    OS << "#ifdef " << Name << "\n"
       << "#undef " << Name << "\n\n";
  }
  ~IfDefEmitter() { OS << "\n#endif // " << Name << "\n\n"; }

private:
  std::string Name;
  raw_ostream &OS;
};

// Simple RAII helper for emitting namespace scope. Name can be a single
// namespace (empty for anonymous namespace) or nested namespace.
class NamespaceEmitter {
public:
  NamespaceEmitter(raw_ostream &OS, StringRef Name)
      : Name(trim(Name).str()), OS(OS) {
    OS << "namespace " << this->Name << " {\n";
  }

  ~NamespaceEmitter() { close(); }

  // Explicit function to close the namespace scopes.
  void close() {
    if (!Closed)
      OS << "} // namespace " << Name << "\n";
    Closed = true;
  }

private:
  // Trim "::" prefix. If the namespace specified is ""::mlir::toy", then the
  // generated namespace scope needs to use
  //
  // namespace mlir::toy {
  // }
  //
  // and cannot use "namespace ::mlir::toy".
  static StringRef trim(StringRef Name) {
    Name.consume_front("::");
    return Name;
  }
  std::string Name;
  raw_ostream &OS;
  bool Closed = false;
};

} // end namespace llvm

#endif // LLVM_TABLEGEN_CODEGENHELPERS_H
