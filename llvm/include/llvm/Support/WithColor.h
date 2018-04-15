//===- WithColor.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_WITHCOLOR_H
#define LLVM_SUPPORT_WITHCOLOR_H

namespace llvm {

class raw_ostream;

// Symbolic names for various syntax elements.
enum class HighlightColor {
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

/// An RAII object that temporarily switches an output stream to a specific
/// color.
class WithColor {
  raw_ostream &OS;
  /// Determine whether colors should be displayed.
  bool colorsEnabled(raw_ostream &OS);

public:
  /// To be used like this: WithColor(OS, HighlightColor::String) << "text";
  WithColor(raw_ostream &OS, HighlightColor S);
  ~WithColor();

  raw_ostream &get() { return OS; }
  operator raw_ostream &() { return OS; }

  /// Convenience method for printing "error: " to stderr.
  static raw_ostream &error();
  /// Convenience method for printing "warning: " to stderr.
  static raw_ostream &warning();
  /// Convenience method for printing "note: " to stderr.
  static raw_ostream &note();

  /// Convenience method for printing "error: " to the given stream.
  static raw_ostream &error(raw_ostream &OS);
  /// Convenience method for printing "warning: " to the given stream.
  static raw_ostream &warning(raw_ostream &OS);
  /// Convenience method for printing "note: " to the given stream.
  static raw_ostream &note(raw_ostream &OS);
};

} // end namespace llvm

#endif // LLVM_LIB_DEBUGINFO_WITHCOLOR_H
