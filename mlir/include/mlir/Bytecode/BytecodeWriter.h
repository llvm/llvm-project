//===- BytecodeWriter.h - MLIR Bytecode Writer ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines interfaces to write MLIR bytecode files/streams.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BYTECODE_BYTECODEWRITER_H
#define MLIR_BYTECODE_BYTECODEWRITER_H

#include "mlir/IR/AsmState.h"

namespace mlir {
class Operation;

/// This class contains the configuration used for the bytecode writer. It
/// controls various aspects of bytecode generation, and contains all of the
/// various bytecode writer hooks.
class BytecodeWriterConfig {
public:
  /// `producer` is an optional string that can be used to identify the producer
  /// of the bytecode when reading. It has no functional effect on the bytecode
  /// serialization.
  BytecodeWriterConfig(StringRef producer = "MLIR" LLVM_VERSION_STRING);
  /// `map` is a fallback resource map, which when provided will attach resource
  /// printers for the fallback resources within the map.
  BytecodeWriterConfig(FallbackAsmResourceMap &map,
                       StringRef producer = "MLIR" LLVM_VERSION_STRING);
  ~BytecodeWriterConfig();

  /// An internal implementation class that contains the state of the
  /// configuration.
  struct Impl;

  /// Return an instance of the internal implementation.
  const Impl &getImpl() const { return *impl; }

  /// Set the desired bytecode version to emit. This function clamps the version
  /// to the existing version if larger than existing. The desired version may
  /// not be used depending on the features used and the actual version required
  /// is returned by bytecode writer entry point.
  void setDesiredBytecodeVersion(int64_t bytecodeVersion);

  //===--------------------------------------------------------------------===//
  // Resources
  //===--------------------------------------------------------------------===//

  /// Attach the given resource printer to the writer configuration.
  void attachResourcePrinter(std::unique_ptr<AsmResourcePrinter> printer);

  /// Attach an resource printer, in the form of a callable, to the
  /// configuration.
  template <typename CallableT>
  std::enable_if_t<std::is_convertible<
      CallableT, function_ref<void(Operation *, AsmResourceBuilder &)>>::value>
  attachResourcePrinter(StringRef name, CallableT &&printFn) {
    attachResourcePrinter(AsmResourcePrinter::fromCallable(
        name, std::forward<CallableT>(printFn)));
  }

  /// Attach resource printers to the AsmState for the fallback resources
  /// in the given map.
  void attachFallbackResourcePrinter(FallbackAsmResourceMap &map) {
    for (auto &printer : map.getPrinters())
      attachResourcePrinter(std::move(printer));
  }

private:
  /// A pointer to allocated storage for the impl state.
  std::unique_ptr<Impl> impl;
};

//===----------------------------------------------------------------------===//
// Entry Points
//===----------------------------------------------------------------------===//

/// Write the bytecode for the given operation to the provided output stream.
/// For streams where it matters, the given stream should be in "binary" mode.
/// It only ever fails if setDesiredByteCodeVersion can't be honored.
LogicalResult writeBytecodeToFile(Operation *op, raw_ostream &os,
                                  const BytecodeWriterConfig &config = {});

} // namespace mlir

#endif // MLIR_BYTECODE_BYTECODEWRITER_H
