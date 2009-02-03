//===-- llvm/Support/Mangler.h - Self-contained name mangler ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Unified name mangler for various backends.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MANGLER_H
#define LLVM_SUPPORT_MANGLER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <string>

namespace llvm {
class Type;
class Module;
class Value;
class GlobalValue;

class Mangler {
  /// Prefix - This string is added to each symbol that is emitted, unless the
  /// symbol is marked as not needing this prefix.
  const char *Prefix;
  
  const char *PrivatePrefix;
  /// UseQuotes - If this is set, the target accepts global names in quotes, 
  /// e.g. "foo bar" is a legal name.  This syntax is used instead of escaping
  /// the space character.  By default, this is false.
  bool UseQuotes;
  
  /// PreserveAsmNames - If this is set, the asm escape character is not removed
  /// from names with 'asm' specifiers. 
  bool PreserveAsmNames;
  
  /// Memo - This is used to remember the name that we assign a value.
  ///
  DenseMap<const Value*, std::string> Memo;

  /// Count - This simple counter is used to unique value names.
  ///
  unsigned Count;
  
  /// TypeMap - If the client wants us to unique types, this keeps track of the
  /// current assignments and TypeCounter keeps track of the next id to assign.
  DenseMap<const Type*, unsigned> TypeMap;
  unsigned TypeCounter;

  /// AcceptableChars - This bitfield contains a one for each character that is
  /// allowed to be part of an unmangled name.
  unsigned AcceptableChars[256/32];
public:

  // Mangler ctor - if a prefix is specified, it will be prepended onto all
  // symbols.
  Mangler(Module &M, const char *Prefix = "", const char *privatePrefix = "");

  /// setUseQuotes - If UseQuotes is set to true, this target accepts quoted
  /// strings for assembler labels.
  void setUseQuotes(bool Val) { UseQuotes = Val; }
  
  /// setPreserveAsmNames - If the mangler should not strip off the asm name
  /// @verbatim identifier (\001), this should be set. @endverbatim
  void setPreserveAsmNames(bool Val) { PreserveAsmNames = Val; }
  
  /// Acceptable Characters - This allows the target to specify which characters
  /// are acceptable to the assembler without being mangled.  By default we
  /// allow letters, numbers, '_', '$', and '.', which is what GAS accepts.
  void markCharAcceptable(unsigned char X) {
    AcceptableChars[X/32] |= 1 << (X&31);
  }
  void markCharUnacceptable(unsigned char X) {
    AcceptableChars[X/32] &= ~(1 << (X&31));
  }
  bool isCharAcceptable(unsigned char X) const {
    return (AcceptableChars[X/32] & (1 << (X&31))) != 0;
  }
  
  /// getValueName - Returns the mangled name of V, an LLVM Value,
  /// in the current module.
  ///
  std::string getValueName(const GlobalValue *V, const char *Suffix = "");
  std::string getValueName(const Value *V);

  /// makeNameProper - We don't want identifier names with ., space, or
  /// - in them, so we mangle these characters into the strings "d_",
  /// "s_", and "D_", respectively. This is a very simple mangling that
  /// doesn't guarantee unique names for values. getValueName already
  /// does this for you, so there's no point calling it on the result
  /// from getValueName.
  ///
  std::string makeNameProper(const std::string &x, const char *Prefix = "");

private:
  /// getTypeID - Return a unique ID for the specified LLVM type.
  ///
  unsigned getTypeID(const Type *Ty);
};

} // End llvm namespace

#endif // LLVM_SUPPORT_MANGLER_H
