//===--- RuntimeDebugBuilder.h --- Helper to insert prints into LLVM-IR ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef RUNTIME_DEBUG_BUILDER_H
#define RUNTIME_DEBUG_BUILDER_H

#include "polly/CodeGen/IRBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace llvm {
class Value;
class Function;
} // namespace llvm

namespace polly {

/// Insert function calls that print certain LLVM values at run time.
///
/// This class inserts libc function calls to print certain LLVM values at
/// run time.
struct RuntimeDebugBuilder {

  /// Generate a constant string into the builder's llvm::Module which can be
  /// passed to createCPUPrinter().
  ///
  /// @param Builder The builder used to emit the printer calls.
  /// @param Str     The string to be printed.

  /// @return        A global containing @p Str.
  static llvm::Value *getPrintableString(PollyIRBuilder &Builder,
                                         llvm::StringRef Str);

  /// Return whether an llvm::Value of the type @p Ty is printable for
  /// debugging.
  ///
  /// That is, whether such a value can be passed to createGPUPrinter()
  /// to be dumped as runtime.  If false is returned, those
  /// functions will fail.
  static bool isPrintable(llvm::Type *Ty);

  /// Print a set of LLVM-IR Values or StringRefs via printf
  ///
  ///  This function emits a call to printf that will print the given arguments.
  ///  It is useful for debugging CPU programs. All arguments given in this list
  ///  will be automatically concatenated and the resulting string will be
  ///  printed atomically. We also support ArrayRef arguments, which can be used
  ///  to provide of id values.
  ///
  ///  @param Builder The builder used to emit the printer calls.
  ///  @param Args    The list of values to print.
  template <typename... Args>
  static void createCPUPrinter(PollyIRBuilder &Builder, Args... args) {
    std::vector<llvm::Value *> Vector;
    createPrinter(Builder, Vector, args...);
  }

private:
  /// Handle Values.
  template <typename... Args>
  static void createPrinter(PollyIRBuilder &Builder,
                            std::vector<llvm::Value *> &Values,
                            llvm::Value *Value, Args... args) {
    Values.push_back(Value);
    createPrinter(Builder, Values, args...);
  }

  /// Handle StringRefs.
  template <typename... Args>
  static void createPrinter(PollyIRBuilder &Builder,
                            std::vector<llvm::Value *> &Values,
                            llvm::StringRef String, Args... args) {
    Values.push_back(getPrintableString(Builder, String));
    createPrinter(Builder, Values, args...);
  }

  /// Handle ArrayRefs.
  template <typename... Args>
  static void createPrinter(PollyIRBuilder &Builder,
                            std::vector<llvm::Value *> &Values,
                            llvm::ArrayRef<llvm::Value *> Array, Args... args) {
    Values.insert(Values.end(), Array.begin(), Array.end());
    createPrinter(Builder, Values, args...);
  }

  /// Print a list of Values.
  static void createPrinter(PollyIRBuilder &Builder,
                            llvm::ArrayRef<llvm::Value *> Values);

  /// Print a list of Values on a CPU.
  static void createCPUPrinterT(PollyIRBuilder &Builder,
                                llvm::ArrayRef<llvm::Value *> Values);

  /// Get a reference to the 'printf' function.
  ///
  /// If the current module does not yet contain a reference to printf, we
  /// insert a reference to it. Otherwise the existing reference is returned.
  static llvm::Function *getPrintF(PollyIRBuilder &Builder);

  /// Call printf
  ///
  /// @param Builder The builder used to insert the code.
  /// @param Format  The format string.
  /// @param Values  The set of values to print.
  static void createPrintF(PollyIRBuilder &Builder, std::string Format,
                           llvm::ArrayRef<llvm::Value *> Values);

  /// Get (and possibly insert) a vprintf declaration into the module.
  static llvm::Function *getVPrintF(PollyIRBuilder &Builder);

  /// Call fflush
  ///
  /// @parma Builder The builder used to insert the code.
  static void createFlush(PollyIRBuilder &Builder);
};
} // namespace polly

extern bool PollyDebugPrinting;

#endif
