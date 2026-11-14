//===- LLVMOps.h - LLVM operation implementation helpers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_LLVMIR_IR_LLVMOPS_H
#define MLIR_LIB_DIALECT_LLVMIR_IR_LLVMOPS_H

#include "mlir/Dialect/LLVMIR/LLVMDialectDecl.h"
#include "mlir/IR/OpImplementation.h"

#include <optional>

namespace mlir::LLVM {
class GEPOp;
class IndirectBrOp;
class SwitchOp;

mlir::Type getI1SameShape(mlir::Type type);
mlir::LLVM::LLVMStructType getValAndBoolStructType(mlir::Type valType);

mlir::ParseResult parseCmpPredicate(mlir::OpAsmParser &parser,
                                    mlir::LLVM::ICmpPredicateAttr &predicate);
mlir::ParseResult parseCmpPredicate(mlir::OpAsmParser &parser,
                                    mlir::LLVM::FCmpPredicateAttr &predicate);
void printCmpPredicate(mlir::OpAsmPrinter &printer, mlir::Operation *,
                       mlir::LLVM::ICmpPredicateAttr predicate);
void printCmpPredicate(mlir::OpAsmPrinter &printer, mlir::Operation *,
                       mlir::LLVM::FCmpPredicateAttr predicate);

void printLLVMLinkage(mlir::OpAsmPrinter &p, mlir::Operation *,
                      mlir::LLVM::LinkageAttr val);
mlir::ParseResult parseLLVMLinkage(mlir::OpAsmParser &p,
                                   mlir::LLVM::LinkageAttr &val);

void printOpBundles(mlir::OpAsmPrinter &p, mlir::Operation *op,
                    mlir::OperandRangeRange opBundleOperands,
                    mlir::TypeRangeRange opBundleOperandTypes,
                    std::optional<mlir::ArrayAttr> opBundleTags);
std::optional<mlir::ParseResult> parseOpBundles(
    mlir::OpAsmParser &p,
    mlir::SmallVector<mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand>>
        &opBundleOperands,
    mlir::SmallVector<mlir::SmallVector<mlir::Type>> &opBundleOperandTypes,
    mlir::ArrayAttr &opBundleTags);

mlir::ParseResult parseSwitchOpCases(
    mlir::OpAsmParser &parser, mlir::Type flagType,
    mlir::DenseIntElementsAttr &caseValues,
    mlir::SmallVectorImpl<mlir::Block *> &caseDestinations,
    mlir::SmallVectorImpl<
        mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand>> &caseOperands,
    mlir::SmallVectorImpl<mlir::SmallVector<mlir::Type>> &caseOperandTypes);
void printSwitchOpCases(mlir::OpAsmPrinter &p, mlir::LLVM::SwitchOp op,
                        mlir::Type flagType,
                        mlir::DenseIntElementsAttr caseValues,
                        mlir::SuccessorRange caseDestinations,
                        mlir::OperandRangeRange caseOperands,
                        const mlir::TypeRangeRange &caseOperandTypes);

mlir::ParseResult parseGEPIndices(
    mlir::OpAsmParser &parser,
    mlir::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &indices,
    mlir::DenseI32ArrayAttr &rawConstantIndices);
void printGEPIndices(mlir::OpAsmPrinter &printer, mlir::LLVM::GEPOp gepOp,
                     mlir::OperandRange indices,
                     mlir::DenseI32ArrayAttr rawConstantIndices);

mlir::ParseResult parseInsertExtractValueElementType(
    mlir::AsmParser &parser, mlir::Type &valueType, mlir::Type containerType,
    mlir::DenseI64ArrayAttr position);
void printInsertExtractValueElementType(mlir::AsmPrinter &printer,
                                        mlir::Operation *op,
                                        mlir::Type valueType,
                                        mlir::Type containerType,
                                        mlir::DenseI64ArrayAttr position);

mlir::ParseResult parseShuffleType(mlir::AsmParser &parser, mlir::Type v1Type,
                                   mlir::Type &resType,
                                   mlir::DenseI32ArrayAttr mask);
void printShuffleType(mlir::AsmPrinter &printer, mlir::Operation *op,
                      mlir::Type v1Type, mlir::Type resType,
                      mlir::DenseI32ArrayAttr mask);

mlir::ParseResult parseIndirectBrOpSucessors(
    mlir::OpAsmParser &parser, mlir::Type &flagType,
    mlir::SmallVectorImpl<mlir::Block *> &succOperandBlocks,
    mlir::SmallVectorImpl<
        mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand>> &succOperands,
    mlir::SmallVectorImpl<mlir::SmallVector<mlir::Type>> &succOperandsTypes);
void printIndirectBrOpSucessors(mlir::OpAsmPrinter &p,
                                mlir::LLVM::IndirectBrOp op,
                                mlir::Type flagType, mlir::SuccessorRange succs,
                                mlir::OperandRangeRange succOperands,
                                const mlir::TypeRangeRange &succOperandsTypes);

//===----------------------------------------------------------------------===//
// Helpers shared by the op partitions
//===----------------------------------------------------------------------===//

/// Returns the attributes of `op` to print: its discardable attributes and
/// all its inherent attributes.
NamedAttrList getAttrsForPrinting(Operation *op);

/// Returns the attributes of `op` to print: its discardable attributes and the
/// inherent attributes `inherentAttrNames`.
NamedAttrList getAttrsForPrinting(Operation *op,
                                  ArrayRef<StringAttr> inherentAttrNames);

/// Returns the elemental type of any LLVM-compatible vector type or self.
Type extractVectorElementType(Type type);

/// Conditions the deletion of the operation to the removal of all its uses.
bool forwardToUsers(Operation *op,
                    SmallVectorImpl<OpOperand *> &newBlockingUses);

// Parses one of the keywords provided in the list `keywords` and returns the
// position of the parsed keyword in the list. If none of the keywords from the
// list is parsed, returns -1.
int parseOptionalKeywordAlternative(OpAsmParser &parser,
                                    ArrayRef<StringRef> keywords);

namespace detail {
using cconv::getMaxEnumValForCConv;
using linkage::getMaxEnumValForLinkage;
using tailcallkind::getMaxEnumValForTailCallKind;

template <typename Ty>
struct EnumTraits {};

#define REGISTER_ENUM_TYPE(Ty)                                                 \
  template <>                                                                  \
  struct EnumTraits<Ty> {                                                      \
    static StringRef stringify(Ty value) { return stringify##Ty(value); }      \
    static unsigned getMaxEnumVal() { return getMaxEnumValFor##Ty(); }         \
  }

REGISTER_ENUM_TYPE(Linkage);
REGISTER_ENUM_TYPE(UnnamedAddr);
REGISTER_ENUM_TYPE(CConv);
REGISTER_ENUM_TYPE(TailCallKind);
REGISTER_ENUM_TYPE(ThreadLocalMode);
REGISTER_ENUM_TYPE(Visibility);

#undef REGISTER_ENUM_TYPE
} // namespace detail

/// Parse an enum from the keyword, or default to the provided default value.
/// The return type is the enum type by default, unless overridden with the
/// second template argument.
template <typename EnumTy, typename RetTy = EnumTy>
RetTy parseOptionalLLVMKeyword(OpAsmParser &parser, EnumTy defaultValue) {
  SmallVector<StringRef, 10> names;
  for (unsigned i = 0, e = detail::EnumTraits<EnumTy>::getMaxEnumVal(); i <= e;
       ++i)
    names.push_back(
        detail::EnumTraits<EnumTy>::stringify(static_cast<EnumTy>(i)));

  int index = parseOptionalKeywordAlternative(parser, names);
  if (index == -1)
    return static_cast<RetTy>(defaultValue);
  return static_cast<RetTy>(index);
}

/// Verifies the operand bundles of `op`.
template <typename OpType>
LogicalResult verifyOperandBundles(OpType &op) {
  OperandRangeRange opBundleOperands = op.getOpBundleOperands();
  std::optional<ArrayAttr> opBundleTags = op.getOpBundleTags();

  auto isStringAttr = [](Attribute tagAttr) {
    return isa<StringAttr>(tagAttr);
  };
  if (opBundleTags && !llvm::all_of(*opBundleTags, isStringAttr))
    return op.emitError("operand bundle tag must be a StringAttr");

  size_t numOpBundles = opBundleOperands.size();
  size_t numOpBundleTags = opBundleTags ? opBundleTags->size() : 0;
  if (numOpBundles != numOpBundleTags)
    return op.emitError("expected ")
           << numOpBundles << " operand bundle tags, but actually got "
           << numOpBundleTags;

  return success();
}
} // namespace mlir::LLVM

#endif // MLIR_LIB_DIALECT_LLVMIR_IR_LLVMOPS_H
