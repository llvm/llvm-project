//===- LvlTypeParser.h - `DimLevelType` parser ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_LVLTYPEPARSER_H
#define MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_LVLTYPEPARSER_H

#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace sparse_tensor {
namespace ir_detail {

//===----------------------------------------------------------------------===//
// These macros are for generating a C++ expression of type
// `std::initializer_list<std::pair<StringRef,DimLevelType>>` since there's
// no way to construct an object of that type directly via C++ code.
#define FOREVERY_LEVELTYPE(DO)                                                 \
  DO(DimLevelType::Dense)                                                      \
  DO(DimLevelType::Compressed)                                                 \
  DO(DimLevelType::CompressedNu)                                               \
  DO(DimLevelType::CompressedNo)                                               \
  DO(DimLevelType::CompressedNuNo)                                             \
  DO(DimLevelType::Singleton)                                                  \
  DO(DimLevelType::SingletonNu)                                                \
  DO(DimLevelType::SingletonNo)                                                \
  DO(DimLevelType::SingletonNuNo)                                              \
  DO(DimLevelType::CompressedWithHi)                                           \
  DO(DimLevelType::CompressedWithHiNu)                                         \
  DO(DimLevelType::CompressedWithHiNo)                                         \
  DO(DimLevelType::CompressedWithHiNuNo)                                       \
  DO(DimLevelType::TwoOutOfFour)
#define LEVELTYPE_INITLIST_ELEMENT(lvlType)                                    \
  std::make_pair(StringRef(toMLIRString(lvlType)), lvlType),
#define LEVELTYPE_INITLIST                                                     \
  { FOREVERY_LEVELTYPE(LEVELTYPE_INITLIST_ELEMENT) }

// TODO(wrengr): Since this parser is non-trivial to construct, is there
// any way to hook into the parsing process so that we construct it only once
// at the begining of parsing and then destroy it once parsing has finished?
class LvlTypeParser {
  const llvm::StringMap<DimLevelType> map;

public:
  explicit LvlTypeParser() : map(LEVELTYPE_INITLIST) {}
#undef LEVELTYPE_INITLIST
#undef LEVELTYPE_INITLIST_ELEMENT
#undef FOREVERY_LEVELTYPE

  std::optional<DimLevelType> lookup(StringRef str) const;
  std::optional<DimLevelType> lookup(StringAttr str) const;
  ParseResult parseLvlType(AsmParser &parser, DimLevelType &out) const;
  FailureOr<DimLevelType> parseLvlType(AsmParser &parser) const;
  // TODO(wrengr): `parseOptionalLvlType`?
  // TODO(wrengr): `parseLvlTypeList`?
};

} // namespace ir_detail
} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_LVLTYPEPARSER_H
