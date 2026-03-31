//===- OpDocGen.h - Op/dialect documentation generator ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions for generating documentation for MLIR dialects,
// operations, attributes, types, and enums from TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_OPDOCGEN_H
#define MLIR_TABLEGEN_GENERATORS_OPDOCGEN_H

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Dialect.h"
#include "mlir/TableGen/EnumInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class Record;
class RecordKeeper;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// A group of operations that share a documentation section.
struct OpDocGroup {
  const Dialect &getDialect() const { return ops.front().getDialect(); }

  /// Summary description of the section.
  std::string summary = "";

  /// Description of the section.
  llvm::StringRef description = "";

  /// Instances inside the section.
  std::vector<Operator> ops;
};

/// Holds all records collected from a dialect relevant for documentation
/// generation.
struct DialectRecords {
  DialectRecords(Dialect dialect, llvm::StringRef inputFilename)
      : dialect(dialect), inputFilename(inputFilename) {}

  Dialect dialect;
  std::string inputFilename;
  std::vector<Attribute> attributes;
  std::vector<AttrDef> attrDefs;
  std::vector<OpDocGroup> ops;
  std::vector<Type> types;
  std::vector<TypeDef> typeDefs;
  std::vector<EnumInfo> enums;
};

/// Collect, filter, and organize all records relevant for dialect documentation
/// generation. opDefs are the op definitions to include (e.g. filtered by
/// the caller). dialect is the dialect to collect records for.
/// keepOpSourceOrder disables alphabetical sorting of ops.
std::optional<DialectRecords>
collectRecords(const llvm::RecordKeeper &records,
               llvm::ArrayRef<const llvm::Record *> opDefs,
               const Dialect &dialect, bool keepOpSourceOrder);

/// Emit documentation for a single operation. stripPrefix is stripped from
/// the fully qualified class name. allowHugoSpecificFeatures enables
/// Hugo-specific markup in attribute descriptions.
void emitOpDoc(const Operator &op, llvm::StringRef stripPrefix,
               bool allowHugoSpecificFeatures, llvm::raw_ostream &os);

/// Emit operation documentation for all ops in records.
bool emitOpDoc(const DialectRecords &records, llvm::StringRef stripPrefix,
               bool allowHugoSpecificFeatures, llvm::raw_ostream &os);

/// Emit attribute definition documentation for all attrDefs in records.
bool emitAttrDefDoc(const DialectRecords &records, llvm::raw_ostream &os);

/// Emit type definition documentation for all typeDefs in records.
bool emitTypeDefDoc(const DialectRecords &records, llvm::raw_ostream &os);

/// Emit enum documentation for all enums in records.
bool emitEnumDoc(const DialectRecords &records, llvm::raw_ostream &os);

/// Emit full dialect documentation including all ops, attrs, types, and enums.
bool emitDialectDoc(const DialectRecords &records, llvm::StringRef stripPrefix,
                    bool allowHugoSpecificFeatures, llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_OPDOCGEN_H
