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
// Simple RAII helper for emitting ifdef-undef-endif scope. `LateUndef` controls
// whether the undef is emitted at the start of the scope (false) or at the end
// of the scope (true).
class IfDefEmitter {
public:
  IfDefEmitter(raw_ostream &OS, StringRef Name, bool LateUndef = false)
      : Name(Name.str()), OS(OS), LateUndef(LateUndef) {
    OS << "#ifdef " << Name << "\n";
    if (!LateUndef)
      OS << "#undef " << Name << "\n";
    OS << "\n";
  }
  ~IfDefEmitter() { close(); }

  // Explicit function to close the ifdef scopes.
  void close() {
    if (Closed)
      return;

    OS << "\n";
    if (LateUndef)
      OS << "#undef " << Name << "\n";
    OS << "#endif // " << Name << "\n\n";
    Closed = true;
  }

private:
  std::string Name;
  raw_ostream &OS;
  bool LateUndef;
  bool Closed = false;
};

// Simple RAII helper for emitting header include guard (ifndef-define-endif).
class IncludeGuardEmitter {
public:
  IncludeGuardEmitter(raw_ostream &OS, StringRef Name)
      : Name(Name.str()), OS(OS) {
    OS << "#ifndef " << Name << "\n"
       << "#define " << Name << "\n\n";
  }
  ~IncludeGuardEmitter() { close(); }

  // Explicit function to close the ifdef scopes.
  void close() {
    if (Closed)
      return;
    OS << "\n#endif // " << Name << "\n\n";
    Closed = true;
  }

private:
  std::string Name;
  raw_ostream &OS;
  bool Closed = false;
};

// Simple RAII helper for emitting namespace scope. Name can be a single
// namespace or nested namespace. If the name is empty, will not generate any
// namespace scope.
class NamespaceEmitter {
public:
  NamespaceEmitter(raw_ostream &OS, StringRef NameUntrimmed)
      : Name(trim(NameUntrimmed).str()), OS(OS) {
    if (!Name.empty())
      OS << "namespace " << Name << " {\n";
  }

  ~NamespaceEmitter() { close(); }

  // Explicit function to close the namespace scopes.
  void close() {
    if (Closed)
      return;
    if (!Name.empty())
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
