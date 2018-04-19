//===--- RuntimeDebugBuilder.h --- Helper to insert prints into LLVM-IR ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
    createPrinter(Builder, /* CPU */ false, Vector, args...);
  }

  /// Print a set of LLVM-IR Values or StringRefs on an NVIDIA GPU.
  ///
  ///  This function emits a call to vprintf that will print the given
  ///  arguments from within a kernel thread. It is useful for debugging
  ///  CUDA program kernels. All arguments given in this list will be
  ///  automatically concatenated and the resulting string will be printed
  ///  atomically. We also support ArrayRef arguments, which can be used to
  ///  provide for example a list of thread-id values.
  ///
  ///  @param Builder The builder used to emit the printer calls.
  ///  @param Args    The list of values to print.
  template <typename... Args>
  static void createGPUPrinter(PollyIRBuilder &Builder, Args... args) {
    std::vector<llvm::Value *> Vector;
    createPrinter(Builder, /* GPU */ true, Vector, args...);
  }

private:
  /// Handle Values.
  template <typename... Args>
  static void createPrinter(PollyIRBuilder &Builder, bool UseGPU,
                            std::vector<llvm::Value *> &Values,
                            llvm::Value *Value, Args... args) {
    Values.push_back(Value);
    createPrinter(Builder, UseGPU, Values, args...);
  }

  /// Handle StringRefs.
  template <typename... Args>
  static void createPrinter(PollyIRBuilder &Builder, bool UseGPU,
                            std::vector<llvm::Value *> &Values,
                            llvm::StringRef String, Args... args) {
    Values.push_back(Builder.CreateGlobalStringPtr(String, "", 4));
    createPrinter(Builder, UseGPU, Values, args...);
  }

  /// Handle ArrayRefs.
  template <typename... Args>
  static void createPrinter(PollyIRBuilder &Builder, bool UseGPU,
                            std::vector<llvm::Value *> &Values,
                            llvm::ArrayRef<llvm::Value *> Array, Args... args) {
    Values.insert(Values.end(), Array.begin(), Array.end());
    createPrinter(Builder, UseGPU, Values, args...);
  }

  /// Print a list of Values.
  static void createPrinter(PollyIRBuilder &Builder, bool UseGPU,
                            llvm::ArrayRef<llvm::Value *> Values);

  /// Print a list of Values on a GPU.
  static void createGPUPrinterT(PollyIRBuilder &Builder,
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

  /// Get (and possibly insert) a NVIDIA address space cast call.
  static llvm::Function *getAddressSpaceCast(PollyIRBuilder &Builder,
                                             unsigned Src, unsigned Dst,
                                             unsigned SrcBits = 8,
                                             unsigned DstBits = 8);

  /// Get identifiers that describe the currently executed GPU thread.
  ///
  /// The result will be a vector that if passed to the GPU printer will result
  /// into a string (initialized to values corresponding to the printing
  /// thread):
  ///
  ///   "> block-id: bidx bid1y bidz | thread-id: tidx tidy tidz "
  static std::vector<llvm::Value *>
  getGPUThreadIdentifiers(PollyIRBuilder &Builder);
};
} // namespace polly

extern bool PollyDebugPrinting;

#endif
