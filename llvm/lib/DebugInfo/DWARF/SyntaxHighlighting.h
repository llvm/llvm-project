//===- SyntaxHighlighting.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_SYNTAXHIGHLIGHTING_H
#define LLVM_LIB_DEBUGINFO_SYNTAXHIGHLIGHTING_H

namespace llvm {

class raw_ostream;

namespace dwarf {
namespace syntax {

// Symbolic names for various syntax elements.
enum HighlightColor {
  Address,
  String,
  Tag,
  Attribute,
  Enumerator,
  Macro,
  Error,
  Warning,
  Note
};

/// An RAII object that temporarily switches an output stream to a
/// specific color.
class WithColor {
  raw_ostream &OS;

public:
  /// To be used like this: WithColor(OS, syntax::String) << "text";
  WithColor(raw_ostream &OS, enum HighlightColor Type);
  ~WithColor();

  raw_ostream &get() { return OS; }
  operator raw_ostream &() { return OS; }
};

} // end namespace syntax
} // end namespace dwarf

} // end namespace llvm

#endif // LLVM_LIB_DEBUGINFO_SYNTAXHIGHLIGHTING_H
