//===- AttrOrTypeFormatGen.h - MLIR attribute and type format generator ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_ATTRORTYPEFORMATGEN_H_
#define MLIR_TOOLS_MLIRTBLGEN_ATTRORTYPEFORMATGEN_H_

#include "mlir/TableGen/Class.h"

namespace mlir {
namespace tblgen {
class AttrOrTypeDef;

/// Generate a parser and printer based on a custom assembly format for an
/// attribute or type.
void generateAttrOrTypeFormat(const AttrOrTypeDef &def, MethodBody &parser,
                              MethodBody &printer);

/// Find all the AttrOrTypeDef for the specified dialect. If no dialect
/// specified and can only find one dialect's defs, use that.
void collectAllDefs(StringRef selectedDialect,
                    ArrayRef<const llvm::Record *> records,
                    SmallVectorImpl<AttrOrTypeDef> &resultDefs);

/// This struct is the base generator used when processing tablegen interfaces.
class DefGenerator {
public:
  virtual ~DefGenerator() = default;
  virtual bool emitDecls(StringRef selectedDialect);
  virtual bool emitDefs(StringRef selectedDialect);

protected:
  DefGenerator(ArrayRef<const llvm::Record *> defs, raw_ostream &os,
               StringRef defType, StringRef valueType, bool isAttrGenerator)
      : defRecords(defs), os(os), defType(defType), valueType(valueType),
        isAttrGenerator(isAttrGenerator) {
    // Sort by occurrence in file.
    llvm::sort(defRecords,
               [](const llvm::Record *lhs, const llvm::Record *rhs) {
                 return lhs->getID() < rhs->getID();
               });
  }

  /// Emit the list of def type names.
  void emitTypeDefList(ArrayRef<AttrOrTypeDef> defs);
  /// Emit the code to dispatch between different defs during parsing/printing.
  void emitParsePrintDispatch(ArrayRef<AttrOrTypeDef> defs);

  /// The set of def records to emit.
  std::vector<const llvm::Record *> defRecords;
  /// The attribute or type class to emit.
  /// The stream to emit to.
  raw_ostream &os;
  /// The prefix of the tablegen def name, e.g. Attr or Type.
  StringRef defType;
  /// The C++ base value type of the def, e.g. Attribute or Type.
  StringRef valueType;
  /// Flag indicating if this generator is for Attributes. False if the
  /// generator is for types.
  bool isAttrGenerator;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_ATTRORTYPEFORMATGEN_H_
