//===- Pass.h - TableGen pass definitions -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PASS_H_
#define MLIR_TABLEGEN_PASS_H_

#include "mlir/Support/LLVM.h"
#include <vector>

namespace llvm {
class Record;
} // namespace llvm

namespace mlir {
namespace tblgen {
//===----------------------------------------------------------------------===//
// PassOption
//===----------------------------------------------------------------------===//
class PassOption {
public:
  explicit PassOption(const llvm::Record *def) : def(def) {}

  /// Return the name for the C++ option variable.
  StringRef getCppVariableName() const;

  /// Return the command line argument to use for this option.
  StringRef getArgument() const;

  /// Return the C++ type of the option.
  StringRef getType() const;

  /// Return the default value of the option.
  std::optional<StringRef> getDefaultValue() const;

  /// Return the description for this option.
  StringRef getDescription() const;

  /// Return the additional flags passed to the option constructor.
  std::optional<StringRef> getAdditionalFlags() const;

  /// Flag indicating if this is a list option.
  bool isListOption() const;

private:
  const llvm::Record *def;
};

//===----------------------------------------------------------------------===//
// PassStatistic
//===----------------------------------------------------------------------===//
class PassStatistic {
public:
  explicit PassStatistic(const llvm::Record *def) : def(def) {}

  /// Return the name for the C++ statistic variable.
  StringRef getCppVariableName() const;

  /// Return the name of the statistic.
  StringRef getName() const;

  /// Return the description for this statistic.
  StringRef getDescription() const;

private:
  const llvm::Record *def;
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Wrapper class providing helper methods for Passes defined in TableGen.
class Pass {
public:
  explicit Pass(const llvm::Record *def);

  /// Return the command line argument of the pass.
  StringRef getArgument() const;

  /// Return the name for the C++ base class.
  StringRef getBaseClass() const;

  /// Return the short 1-line summary of the pass.
  StringRef getSummary() const;

  /// Return the description of the pass.
  StringRef getDescription() const;

  /// Return the C++ constructor call to create an instance of this pass.
  StringRef getConstructor() const;

  /// Return the dialects this pass needs to be registered.
  ArrayRef<StringRef> getDependentDialects() const;

  /// Return the options provided by this pass.
  ArrayRef<PassOption> getOptions() const;

  /// Return the statistics provided by this pass.
  ArrayRef<PassStatistic> getStatistics() const;

  const llvm::Record *getDef() const { return def; }

private:
  const llvm::Record *def;
  std::vector<StringRef> dependentDialects;
  std::vector<PassOption> options;
  std::vector<PassStatistic> statistics;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_PASS_H_
