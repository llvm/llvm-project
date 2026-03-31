//===- DialectInterfacesGen.h - Dialect interface generator -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_DIALECTINTERFACESGEN_H
#define MLIR_TABLEGEN_GENERATORS_DIALECTINTERFACESGEN_H

#include "mlir/TableGen/Interfaces.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

namespace llvm {
class Record;
class RecordKeeper;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Get all DialectInterface definitions from the given records, excluding those
/// defined outside the top-level file.
std::vector<const llvm::Record *>
getAllDialectInterfaceDefinitions(const llvm::RecordKeeper &records);

//===----------------------------------------------------------------------===//
// DialectInterfaceGenerator
//===----------------------------------------------------------------------===//

/// Generator for dialect interface declarations from TableGen records.
class DialectInterfaceGenerator {
public:
  DialectInterfaceGenerator(const llvm::RecordKeeper &records,
                            llvm::raw_ostream &os);

  virtual ~DialectInterfaceGenerator() = default;

  virtual bool emitInterfaceDecls();

protected:
  virtual void emitInterfaceDecl(const DialectInterface &interface);

  /// The set of interface records to emit.
  std::vector<const llvm::Record *> defs;
  /// The stream to emit to.
  llvm::raw_ostream &os;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_DIALECTINTERFACESGEN_H
