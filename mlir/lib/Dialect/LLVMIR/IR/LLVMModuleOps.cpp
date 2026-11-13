//===- LLVMModuleOps.cpp - LLVM dialect module-level ops ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Module` of the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMModuleOps.h"

#include "IR/LLVMOps.h"
#include "mlir/Dialect/LLVMIR/LLVMConstantOps.h"
#include "mlir/Dialect/LLVMIR/LLVMControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMMemoryOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

using namespace mlir;
using namespace mlir::LLVM;

NamedAttrList mlir::LLVM::getAttrsForPrinting(Operation *op) {
  NamedAttrList attrs(op->getRawDictionaryAttrs());
  op->getName().walkInherentAttrs(
      op, [&](StringRef name, Attribute &attr) { attrs.set(name, attr); });
  return attrs;
}

// TODO: Split inherent attributes from discardable attributes in the assembly
// syntax of GlobalOp and AliasOp. For now, print the selected inherent
// attributes in the regular attribute dictionary.
NamedAttrList
mlir::LLVM::getAttrsForPrinting(Operation *op,
                                ArrayRef<StringAttr> inherentAttrNames) {
  NamedAttrList attrs(op->getRawDictionaryAttrs());
  for (StringAttr name : inherentAttrNames)
    if (std::optional<Attribute> attr = op->getInherentAttr(name);
        attr && *attr)
      attrs.set(name, *attr);
  return attrs;
}

/// Verifies `symbol`'s use in `op` to ensure the symbol is a valid and
/// fully defined llvm.func.
static LogicalResult verifySymbolAttrUse(FlatSymbolRefAttr symbol,
                                         Operation *op,
                                         SymbolTableCollection &symbolTable) {
  StringRef name = symbol.getValue();
  auto func =
      symbolTable.lookupNearestSymbolFrom<LLVMFuncOp>(op, symbol.getAttr());
  if (!func)
    return op->emitOpError("'")
           << name << "' does not reference a valid LLVM function";
  if (func.isExternal())
    return op->emitOpError("'") << name << "' does not have a definition";
  return success();
}

// Parses one of the keywords provided in the list `keywords` and returns the
// position of the parsed keyword in the list. If none of the keywords from the
// list is parsed, returns -1.
int mlir::LLVM::parseOptionalKeywordAlternative(OpAsmParser &parser,
                                                ArrayRef<StringRef> keywords) {
  for (const auto &en : llvm::enumerate(keywords)) {
    if (succeeded(parser.parseOptionalKeyword(en.value())))
      return en.index();
  }
  return -1;
}

void mlir::LLVM::printLLVMLinkage(OpAsmPrinter &p, Operation *,
                                  LinkageAttr val) {
  p << stringifyLinkage(val.getLinkage());
}

ParseResult mlir::LLVM::parseLLVMLinkage(OpAsmParser &p, LinkageAttr &val) {
  val = LinkageAttr::get(
      p.getContext(),
      parseOptionalLLVMKeyword<LLVM::Linkage>(p, LLVM::Linkage::External));
  return success();
}

//===----------------------------------------------------------------------===//
// Verifier for LLVM::ComdatOp.
//===----------------------------------------------------------------------===//

void ComdatOp::build(OpBuilder &builder, OperationState &result,
                     StringRef symName) {
  result.addAttribute(getSymNameAttrName(result.name),
                      builder.getStringAttr(symName));
  Region *body = result.addRegion();
  body->emplaceBlock();
}

LogicalResult ComdatOp::verifyRegions() {
  Region &body = getBody();
  for (Operation &op : body.getOps())
    if (!isa<ComdatSelectorOp>(op))
      return op.emitError(
          "only comdat selector symbols can appear in a comdat region");

  return success();
}

//===----------------------------------------------------------------------===//
// Builder, printer and verifier for LLVM::GlobalOp.
//===----------------------------------------------------------------------===//

void GlobalOp::build(OpBuilder &builder, OperationState &result, Type type,
                     bool isConstant, Linkage linkage, StringRef name,
                     Attribute value, uint64_t alignment, unsigned addrSpace,
                     bool dsoLocal, ThreadLocalMode threadModel,
                     SymbolRefAttr comdat, ArrayRef<NamedAttribute> attrs,
                     ArrayRef<Attribute> dbgExprs) {
  result.getOrAddProperties<Properties>().sym_name =
      builder.getStringAttr(name);
  result.addAttribute(getGlobalTypeAttrName(result.name), TypeAttr::get(type));
  result.addAttribute(
      getTlsModeAttrName(result.name),
      ThreadLocalModeAttr::get(builder.getContext(), threadModel));
  if (isConstant)
    result.addAttribute(getConstantAttrName(result.name),
                        builder.getUnitAttr());
  if (value)
    result.addAttribute(getValueAttrName(result.name), value);
  if (dsoLocal)
    result.addAttribute(getDsoLocalAttrName(result.name),
                        builder.getUnitAttr());
  if (comdat)
    result.addAttribute(getComdatAttrName(result.name), comdat);

  // Only add an alignment attribute if the "alignment" input
  // is different from 0. The value must also be a power of two, but
  // this is tested in GlobalOp::verify, not here.
  if (alignment != 0)
    result.addAttribute(getAlignmentAttrName(result.name),
                        builder.getI64IntegerAttr(alignment));

  result.addAttribute(getLinkageAttrName(result.name),
                      LinkageAttr::get(builder.getContext(), linkage));
  if (addrSpace != 0)
    result.addAttribute(getAddrSpaceAttrName(result.name),
                        builder.getI32IntegerAttr(addrSpace));
  result.attributes.append(attrs.begin(), attrs.end());

  if (!dbgExprs.empty())
    result.addAttribute(getDbgExprsAttrName(result.name),
                        ArrayAttr::get(builder.getContext(), dbgExprs));

  result.addRegion();
}

template <typename OpType>
static void printCommonGlobalAndAlias(OpAsmPrinter &p, OpType op) {
  p << ' ' << stringifyLinkage(op.getLinkage()) << ' ';
  StringRef visibility = stringifyVisibility(op.getVisibility_());
  if (!visibility.empty())
    p << visibility << ' ';

  if (ThreadLocalMode mode = op.getTlsMode();
      mode != ThreadLocalMode::NotThreadLocal) {
    p << "thread_local";
    if (mode != ThreadLocalMode::GeneralDynamic)
      p << '(' << mode << ')';
    p << ' ';
  }

  if (auto unnamedAddr = op.getUnnamedAddr()) {
    StringRef str = stringifyUnnamedAddr(*unnamedAddr);
    if (!str.empty())
      p << str << ' ';
  }
}

void GlobalOp::print(OpAsmPrinter &p) {
  printCommonGlobalAndAlias<GlobalOp>(p, *this);
  if (getConstant())
    p << "constant ";
  p.printSymbolName(getSymName());
  p << '(';
  if (auto value = getValueOrNull())
    p.printAttribute(value);
  p << ')';
  if (auto comdat = getComdat())
    p << " comdat(" << *comdat << ')';

  // Note that the alignment attribute is printed using the
  // default syntax here, even though it is an inherent attribute
  // (as defined in https://mlir.llvm.org/docs/LangRef/#attributes)
  p.printOptionalAttrDict(
      getAttrsForPrinting(
          *this, {getDsoLocalAttrName(), getExternallyInitializedAttrName(),
                  getAlignmentAttrName(), getAddrSpaceAttrName(),
                  getSectionAttrName(), getAssociatedAttrName(),
                  getAbsoluteSymbolAttrName(), getDbgExprsAttrName(),
                  getTargetSpecificAttrsAttrName(), getSymVisibilityAttrName()})
          .getAttrs(),
      {getSymNameAttrName(), getGlobalTypeAttrName(), getConstantAttrName(),
       getValueAttrName(), getLinkageAttrName(), getUnnamedAddrAttrName(),
       getTlsModeAttrName(), getVisibility_AttrName(), getComdatAttrName()});

  // Print the trailing type unless it's a string global.
  if (llvm::dyn_cast_or_null<StringAttr>(getValueOrNull()))
    return;
  p << " : " << getType();

  Region &initializer = getInitializerRegion();
  if (!initializer.empty()) {
    p << ' ';
    p.printRegion(initializer, /*printEntryBlockArgs=*/false);
  }
}

static LogicalResult verifyComdat(Operation *op,
                                  std::optional<SymbolRefAttr> attr,
                                  SymbolTableCollection &symbolTable) {
  if (!attr)
    return success();

  auto *comdatSelector = symbolTable.lookupNearestSymbolFrom(op, *attr);
  if (!isa_and_nonnull<ComdatSelectorOp>(comdatSelector))
    return op->emitError() << "expected comdat symbol";

  return success();
}

static LogicalResult verifyBlockTags(LLVMFuncOp funcOp) {
  llvm::DenseSet<BlockTagAttr> blockTags;
  // Note that presence of `BlockTagOp`s currently can't prevent an unrecheable
  // block to be removed by canonicalizer's region simplify pass, which needs to
  // be dialect aware to allow extra constraints to be described.
  WalkResult res = funcOp.walk([&](BlockTagOp blockTagOp) {
    if (blockTags.contains(blockTagOp.getTag())) {
      blockTagOp.emitError()
          << "duplicate block tag '" << blockTagOp.getTag().getId()
          << "' in the same function: ";
      return WalkResult::interrupt();
    }
    blockTags.insert(blockTagOp.getTag());
    return WalkResult::advance();
  });

  return failure(res.wasInterrupted());
}

/// Parse common attributes that might show up in the same order in both
/// GlobalOp and AliasOp.
template <typename OpType>
static ParseResult parseCommonGlobalAndAlias(OpAsmParser &parser,
                                             OperationState &result) {
  MLIRContext *ctx = parser.getContext();

  // Parse optional linkage, default to External.
  result.addAttribute(
      OpType::getLinkageAttrName(result.name),
      LLVM::LinkageAttr::get(ctx, parseOptionalLLVMKeyword<Linkage>(
                                      parser, LLVM::Linkage::External)));

  // Parse optional visibility, default to Default.
  result.addAttribute(OpType::getVisibility_AttrName(result.name),
                      parser.getBuilder().getI64IntegerAttr(
                          parseOptionalLLVMKeyword<LLVM::Visibility, int64_t>(
                              parser, LLVM::Visibility::Default)));

  if (succeeded(parser.parseOptionalKeyword("thread_local"))) {
    ThreadLocalMode threadModel = ThreadLocalMode::GeneralDynamic;

    if (succeeded(parser.parseOptionalLParen())) {
      SMLoc kwLoc;
      if (parser.getCurrentLocation(&kwLoc))
        return failure();
      threadModel = parseOptionalLLVMKeyword<ThreadLocalMode>(
          parser, ThreadLocalMode::NotThreadLocal);
      if (threadModel == ThreadLocalMode::NotThreadLocal) {
        parser.emitError(kwLoc, "invalid value for thread_local");
        return failure();
      }
      if (parser.parseRParen())
        return failure();
    }
    result.addAttribute(OpType::getTlsModeAttrName(result.name),
                        ThreadLocalModeAttr::get(ctx, threadModel));
  }

  // Parse optional UnnamedAddr, default to None.
  result.addAttribute(OpType::getUnnamedAddrAttrName(result.name),
                      parser.getBuilder().getI64IntegerAttr(
                          parseOptionalLLVMKeyword<UnnamedAddr, int64_t>(
                              parser, LLVM::UnnamedAddr::None)));

  return success();
}

// operation ::= `llvm.mlir.global` linkage? visibility?
//               (`unnamed_addr` | `local_unnamed_addr`)?
//               (`thread_local` (`(` tls-mode `)`)? )?
//               `constant`? `@` identifier
//               `(` attribute? `)` (`comdat(` symbol-ref-id `)`)?
//               attribute-list? (`:` type)? region?
//
// The type can be omitted for string attributes, in which case it will be
// inferred from the value of the string as [strlen(value) x i8].
ParseResult GlobalOp::parse(OpAsmParser &parser, OperationState &result) {
  // Call into common parsing between GlobalOp and AliasOp.
  if (parseCommonGlobalAndAlias<GlobalOp>(parser, result).failed())
    return failure();

  if (succeeded(parser.parseOptionalKeyword("constant")))
    result.addAttribute(getConstantAttrName(result.name),
                        parser.getBuilder().getUnitAttr());

  StringAttr name;
  if (parser.parseSymbolName(name, getSymNameAttrName(result.name),
                             result.attributes) ||
      parser.parseLParen())
    return failure();

  Attribute value;
  if (parser.parseOptionalRParen()) {
    if (parser.parseAttribute(value, getValueAttrName(result.name),
                              result.attributes) ||
        parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("comdat"))) {
    SymbolRefAttr comdat;
    if (parser.parseLParen() || parser.parseAttribute(comdat) ||
        parser.parseRParen())
      return failure();

    result.addAttribute(getComdatAttrName(result.name), comdat);
  }

  SmallVector<Type, 1> types;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseOptionalColonTypeList(types))
    return failure();

  if (types.size() > 1)
    return parser.emitError(parser.getNameLoc(), "expected zero or one type");

  Region &initRegion = *result.addRegion();
  if (types.empty()) {
    if (auto strAttr = llvm::dyn_cast_or_null<StringAttr>(value)) {
      MLIRContext *context = parser.getContext();
      auto arrayType = LLVM::LLVMArrayType::get(IntegerType::get(context, 8),
                                                strAttr.getValue().size());
      types.push_back(arrayType);
    } else {
      return parser.emitError(parser.getNameLoc(),
                              "type can only be omitted for string globals");
    }
  } else {
    OptionalParseResult parseResult =
        parser.parseOptionalRegion(initRegion, /*arguments=*/{},
                                   /*argTypes=*/{});
    if (parseResult.has_value() && failed(*parseResult))
      return failure();
  }

  result.addAttribute(getGlobalTypeAttrName(result.name),
                      TypeAttr::get(types[0]));
  return success();
}

static bool isZeroAttribute(Attribute value) {
  if (auto intValue = llvm::dyn_cast<IntegerAttr>(value))
    return intValue.getValue().isZero();
  if (auto fpValue = llvm::dyn_cast<FloatAttr>(value))
    return fpValue.getValue().isZero();
  if (auto splatValue = llvm::dyn_cast<SplatElementsAttr>(value))
    return isZeroAttribute(splatValue.getSplatValue<Attribute>());
  if (auto elementsValue = llvm::dyn_cast<ElementsAttr>(value))
    return llvm::all_of(elementsValue.getValues<Attribute>(), isZeroAttribute);
  if (auto arrayValue = llvm::dyn_cast<ArrayAttr>(value))
    return llvm::all_of(arrayValue.getValue(), isZeroAttribute);
  return false;
}

LogicalResult GlobalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyComdat(*this, getComdat(), symbolTable);
}

LogicalResult GlobalOp::verify() {
  bool validType = isCompatibleOuterType(getType())
                       ? !llvm::isa<LLVMVoidType, TokenType, LLVMMetadataType,
                                    LLVMLabelType>(getType())
                       : llvm::isa<PointerElementTypeInterface>(getType());
  if (!validType)
    return emitOpError(
        "expects type to be a valid element type for an LLVM global");
  if ((*this)->getParentOp() && !satisfiesLLVMModule((*this)->getParentOp()))
    return emitOpError("must appear at the module level");

  if (auto strAttr = llvm::dyn_cast_or_null<StringAttr>(getValueOrNull())) {
    auto type = llvm::dyn_cast<LLVMArrayType>(getType());
    IntegerType elementType =
        type ? llvm::dyn_cast<IntegerType>(type.getElementType()) : nullptr;
    if (!elementType || elementType.getWidth() != 8 ||
        type.getNumElements() != strAttr.getValue().size())
      return emitOpError(
          "requires an i8 array type of the length equal to that of the string "
          "attribute");
  }

  if (auto targetExtType = dyn_cast<LLVMTargetExtType>(getType())) {
    if (!targetExtType.hasProperty(LLVMTargetExtType::CanBeGlobal))
      return emitOpError()
             << "this target extension type cannot be used in a global";

    if (Attribute value = getValueOrNull())
      return emitOpError() << "global with target extension type can only be "
                              "initialized with zero-initializer";
  }

  if (getLinkage() == Linkage::Common) {
    if (Attribute value = getValueOrNull()) {
      if (!isZeroAttribute(value)) {
        return emitOpError()
               << "expected zero value for '"
               << stringifyLinkage(Linkage::Common) << "' linkage";
      }
    }
  }

  if (getLinkage() == Linkage::Appending) {
    if (!llvm::isa<LLVMArrayType>(getType())) {
      return emitOpError() << "expected array type for '"
                           << stringifyLinkage(Linkage::Appending)
                           << "' linkage";
    }
  }

  std::optional<uint64_t> alignAttr = getAlignment();
  if (alignAttr.has_value()) {
    uint64_t value = alignAttr.value();
    if (!llvm::isPowerOf2_64(value))
      return emitError() << "alignment attribute is not a power of 2";
  }

  if (FlatSymbolRefAttr associated = getAssociatedAttr()) {
    if (associated.getValue() == getSymName())
      return emitOpError("associated cannot refer to the global itself");
  }

  if (ArrayAttr absSym = getAbsoluteSymbolAttr()) {
    if (absSym.empty() || absSym.size() % 2 != 0)
      return emitOpError(
          "absolute_symbol must contain one or more integer range pairs");
    Type pairType;
    for (Attribute attr : absSym) {
      auto intAttr = dyn_cast<IntegerAttr>(attr);
      if (!intAttr)
        return emitOpError("absolute_symbol operands must be integers");
      if (!pairType)
        pairType = intAttr.getType();
      else if (intAttr.getType() != pairType)
        return emitOpError("absolute_symbol range pair types must match");
    }
  }

  return success();
}

LogicalResult GlobalOp::verifyRegions() {
  if (Block *b = getInitializerBlock()) {
    ReturnOp ret = cast<ReturnOp>(b->getTerminator());
    if (ret.operand_type_begin() == ret.operand_type_end())
      return emitOpError("initializer region cannot return void");
    if (*ret.operand_type_begin() != getType())
      return emitOpError("initializer region type ")
             << *ret.operand_type_begin() << " does not match global type "
             << getType();

    for (Operation &op : *b) {
      auto iface = dyn_cast<MemoryEffectOpInterface>(op);
      if (!iface || !iface.hasNoEffect())
        return op.emitError()
               << "ops with side effects not allowed in global initializers";
    }

    if (getValueOrNull())
      return emitOpError("cannot have both initializer value and region");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LLVM::GlobalCtorsOp
//===----------------------------------------------------------------------===//

static LogicalResult checkGlobalXtorData(Operation *op, ArrayAttr data) {
  if (data.empty())
    return success();

  if (llvm::all_of(data.getAsRange<Attribute>(), [](Attribute v) {
        return isa<FlatSymbolRefAttr, ZeroAttr>(v);
      }))
    return success();
  return op->emitError("data element must be symbol or #llvm.zero");
}

LogicalResult
GlobalCtorsOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  for (Attribute ctor : getCtors()) {
    if (failed(verifySymbolAttrUse(llvm::cast<FlatSymbolRefAttr>(ctor), *this,
                                   symbolTable)))
      return failure();
  }
  return success();
}

LogicalResult GlobalCtorsOp::verify() {
  if (checkGlobalXtorData(*this, getData()).failed())
    return failure();

  if (getCtors().size() == getPriorities().size() &&
      getCtors().size() == getData().size())
    return success();
  return emitError(
      "ctors, priorities, and data must have the same number of elements");
}

//===----------------------------------------------------------------------===//
// LLVM::GlobalDtorsOp
//===----------------------------------------------------------------------===//

LogicalResult
GlobalDtorsOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  for (Attribute dtor : getDtors()) {
    if (failed(verifySymbolAttrUse(llvm::cast<FlatSymbolRefAttr>(dtor), *this,
                                   symbolTable)))
      return failure();
  }
  return success();
}

LogicalResult GlobalDtorsOp::verify() {
  if (checkGlobalXtorData(*this, getData()).failed())
    return failure();

  if (getDtors().size() == getPriorities().size() &&
      getDtors().size() == getData().size())
    return success();
  return emitError(
      "dtors, priorities, and data must have the same number of elements");
}

//===----------------------------------------------------------------------===//
// Builder, printer and verifier for LLVM::AliasOp.
//===----------------------------------------------------------------------===//

void AliasOp::build(OpBuilder &builder, OperationState &result, Type type,
                    Linkage linkage, StringRef name, bool dsoLocal,
                    ThreadLocalMode threadModel,
                    ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(getSymNameAttrName(result.name),
                      builder.getStringAttr(name));
  result.addAttribute(getAliasTypeAttrName(result.name), TypeAttr::get(type));
  result.addAttribute(
      getTlsModeAttrName(result.name),
      ThreadLocalModeAttr::get(builder.getContext(), threadModel));
  if (dsoLocal)
    result.addAttribute(getDsoLocalAttrName(result.name),
                        builder.getUnitAttr());

  result.addAttribute(getLinkageAttrName(result.name),
                      LinkageAttr::get(builder.getContext(), linkage));
  result.attributes.append(attrs.begin(), attrs.end());

  result.addRegion();
}

void AliasOp::print(OpAsmPrinter &p) {
  printCommonGlobalAndAlias<AliasOp>(p, *this);

  p.printSymbolName(getSymName());
  p.printOptionalAttrDict(
      getAttrsForPrinting(*this,
                          {getDsoLocalAttrName(), getSymVisibilityAttrName()})
          .getAttrs(),
      {getSymNameAttrName(), getAliasTypeAttrName(), getLinkageAttrName(),
       getUnnamedAddrAttrName(), getTlsModeAttrName(),
       getVisibility_AttrName()});

  // Print the trailing type.
  p << " : " << getType() << ' ';
  // Print the initializer region.
  p.printRegion(getInitializerRegion(), /*printEntryBlockArgs=*/false);
}

// operation ::= `llvm.mlir.alias` linkage? visibility?
//               (`unnamed_addr` | `local_unnamed_addr`)?
//               (`thread_local` (`(` tls-mode `)`)? )?
//               `@` identifier `(` attribute? `)`
//               attribute-list? `:` type region
//
ParseResult AliasOp::parse(OpAsmParser &parser, OperationState &result) {
  // Call into common parsing between GlobalOp and AliasOp.
  if (parseCommonGlobalAndAlias<AliasOp>(parser, result).failed())
    return failure();

  StringAttr name;
  if (parser.parseSymbolName(name, getSymNameAttrName(result.name),
                             result.attributes))
    return failure();

  SmallVector<Type, 1> types;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseOptionalColonTypeList(types))
    return failure();

  if (types.size() > 1)
    return parser.emitError(parser.getNameLoc(), "expected zero or one type");

  Region &initRegion = *result.addRegion();
  if (parser.parseRegion(initRegion).failed())
    return failure();

  result.addAttribute(getAliasTypeAttrName(result.name),
                      TypeAttr::get(types[0]));
  return success();
}

LogicalResult AliasOp::verify() {
  bool validType = isCompatibleOuterType(getType())
                       ? !llvm::isa<LLVMVoidType, TokenType, LLVMMetadataType,
                                    LLVMLabelType>(getType())
                       : llvm::isa<PointerElementTypeInterface>(getType());
  if (!validType)
    return emitOpError(
        "expects type to be a valid element type for an LLVM global alias");

  // This matches LLVM IR verification logic, see llvm/lib/IR/Verifier.cpp
  switch (getLinkage()) {
  case Linkage::External:
  case Linkage::Internal:
  case Linkage::Private:
  case Linkage::Weak:
  case Linkage::WeakODR:
  case Linkage::Linkonce:
  case Linkage::LinkonceODR:
  case Linkage::AvailableExternally:
    break;
  default:
    return emitOpError()
           << "'" << stringifyLinkage(getLinkage())
           << "' linkage not supported in aliases, available options: private, "
              "internal, linkonce, weak, linkonce_odr, weak_odr, external or "
              "available_externally";
  }

  return success();
}

LogicalResult AliasOp::verifyRegions() {
  Block &b = getInitializerBlock();
  auto ret = cast<ReturnOp>(b.getTerminator());
  if (ret.getNumOperands() == 0 ||
      !isa<LLVM::LLVMPointerType>(ret.getOperand(0).getType()))
    return emitOpError("initializer region must always return a pointer");

  for (Operation &op : b) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface || !iface.hasNoEffect())
      return op.emitError()
             << "ops with side effects are not allowed in alias initializers";
  }

  return success();
}

unsigned AliasOp::getAddrSpace() {
  Block &initializer = getInitializerBlock();
  auto ret = cast<ReturnOp>(initializer.getTerminator());
  auto ptrTy = cast<LLVMPointerType>(ret.getOperand(0).getType());
  return ptrTy.getAddressSpace();
}

//===----------------------------------------------------------------------===//
// IFuncOp
//===----------------------------------------------------------------------===//

void IFuncOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                    Type iFuncType, StringRef resolverName, Type resolverType,
                    Linkage linkage, LLVM::Visibility visibility) {
  return build(builder, result, name, iFuncType, resolverName, resolverType,
               linkage, /*dso_local=*/false, /*address_space=*/0,
               UnnamedAddr::None, visibility, /*sym_visibility=*/nullptr);
}

LogicalResult IFuncOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *symbol =
      symbolTable.lookupSymbolIn(parentLLVMModule(*this), getResolverAttr());
  // This matches LLVM IR verification logic, see llvm/lib/IR/Verifier.cpp
  auto resolver = dyn_cast_if_present<LLVMFuncOp>(symbol);
  auto alias = dyn_cast_if_present<AliasOp>(symbol);
  while (alias) {
    Block &initBlock = alias.getInitializerBlock();
    auto returnOp = cast<ReturnOp>(initBlock.getTerminator());
    auto addrOp = returnOp.getArg().getDefiningOp<AddressOfOp>();
    // FIXME: This is a best effort solution. The AliasOp body might be more
    // complex and in that case we bail out with success. To completely match
    // the LLVM IR logic it would be necessary to implement proper alias and
    // cast stripping.
    if (!addrOp)
      return success();
    resolver = addrOp.getFunction(symbolTable);
    alias = addrOp.getAlias(symbolTable);
  }
  if (!resolver)
    return emitOpError("must have a function resolver");
  Linkage linkage = resolver.getLinkage();
  if (resolver.isExternal() || linkage == Linkage::AvailableExternally)
    return emitOpError("resolver must be a definition");
  if (!isa<LLVMPointerType>(resolver.getFunctionType().getReturnType()))
    return emitOpError("resolver must return a pointer");
  auto resolverPtr = dyn_cast<LLVMPointerType>(getResolverType());
  if (!resolverPtr || resolverPtr.getAddressSpace() != getAddressSpace())
    return emitOpError("resolver has incorrect type");
  return success();
}

LogicalResult IFuncOp::verify() {
  switch (getLinkage()) {
  case Linkage::External:
  case Linkage::Internal:
  case Linkage::Private:
  case Linkage::Weak:
  case Linkage::WeakODR:
  case Linkage::Linkonce:
  case Linkage::LinkonceODR:
    break;
  default:
    return emitOpError() << "'" << stringifyLinkage(getLinkage())
                         << "' linkage not supported in ifuncs, available "
                            "options: private, internal, linkonce, weak, "
                            "linkonce_odr, weak_odr, or external linkage";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Implementations for LLVM::LLVMFuncOp.
//===----------------------------------------------------------------------===//

// Add the entry block to the function.
Block *LLVMFuncOp::addEntryBlock(OpBuilder &builder) {
  assert(empty() && "function already has an entry block");
  OpBuilder::InsertionGuard g(builder);
  Block *entry = builder.createBlock(&getBody());

  // FIXME: Allow passing in proper locations for the entry arguments.
  LLVMFunctionType type = getFunctionType();
  for (unsigned i = 0, e = type.getNumParams(); i < e; ++i)
    entry->addArgument(type.getParamType(i), getLoc());
  return entry;
}

void LLVMFuncOp::build(OpBuilder &builder, OperationState &result,
                       StringRef name, Type type, LLVM::Linkage linkage,
                       bool dsoLocal, CConv cconv, SymbolRefAttr comdat,
                       ArrayRef<NamedAttribute> attrs,
                       ArrayRef<DictionaryAttr> argAttrs,
                       std::optional<uint64_t> functionEntryCount) {
  result.addRegion();
  result.addAttribute(getSymNameAttrName(result.name),
                      builder.getStringAttr(name));
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));
  result.addAttribute(getLinkageAttrName(result.name),
                      LinkageAttr::get(builder.getContext(), linkage));
  result.addAttribute(getCConvAttrName(result.name),
                      CConvAttr::get(builder.getContext(), cconv));
  result.attributes.append(attrs.begin(), attrs.end());
  if (dsoLocal)
    result.addAttribute(getDsoLocalAttrName(result.name),
                        builder.getUnitAttr());
  if (comdat)
    result.addAttribute(getComdatAttrName(result.name), comdat);
  if (functionEntryCount)
    result.addAttribute(getFunctionEntryCountAttrName(result.name),
                        FunctionEntryCountAttr::get(builder.getContext(),
                                                    *functionEntryCount,
                                                    ArrayRef<uint64_t>{}));
#ifndef NDEBUG
  std::optional<NamedAttribute> duplicate = result.attributes.findDuplicate();
  if (duplicate.has_value()) {
    llvm::report_fatal_error(
        Twine("LLVMFuncOp propagated an attribute that is meant "
              "to be constructed by the builder: ") +
        duplicate->getName().str());
  }
#endif
  if (argAttrs.empty())
    return;

  assert(llvm::cast<LLVMFunctionType>(type).getNumParams() == argAttrs.size() &&
         "expected as many argument attribute lists as arguments");
  call_interface_impl::addArgAndResultAttrs(
      builder, result, argAttrs, /*resultAttrs=*/{},
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

// Builds an LLVM function type from the given lists of input and output types.
// Returns a null type if any of the types provided are non-LLVM types, or if
// there is more than one output type.
static Type
buildLLVMFunctionType(OpAsmParser &parser, SMLoc loc, ArrayRef<Type> inputs,
                      ArrayRef<Type> outputs,
                      function_interface_impl::VariadicFlag variadicFlag) {
  Builder &b = parser.getBuilder();
  if (outputs.size() > 1) {
    parser.emitError(loc, "failed to construct function type: expected zero or "
                          "one function result");
    return {};
  }

  // Convert inputs to LLVM types, exit early on error.
  SmallVector<Type, 4> llvmInputs;
  for (auto t : inputs) {
    if (!isCompatibleType(t)) {
      parser.emitError(loc, "failed to construct function type: expected LLVM "
                            "type for function arguments");
      return {};
    }
    llvmInputs.push_back(t);
  }

  // No output is denoted as "void" in LLVM type system.
  Type llvmOutput =
      outputs.empty() ? LLVMVoidType::get(b.getContext()) : outputs.front();
  if (!isCompatibleType(llvmOutput)) {
    parser.emitError(loc, "failed to construct function type: expected LLVM "
                          "type for function results")
        << llvmOutput;
    return {};
  }
  return LLVMFunctionType::get(llvmOutput, llvmInputs,
                               variadicFlag.isVariadic());
}

// Parses an LLVM function.
//
// operation ::= `llvm.func` linkage? cconv? function-signature
//                (`comdat(` symbol-ref-id `)`)?
//                function-attributes?
//                function-body
//
ParseResult LLVMFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  // Default to external linkage if no keyword is provided.
  result.addAttribute(getLinkageAttrName(result.name),
                      LinkageAttr::get(parser.getContext(),
                                       parseOptionalLLVMKeyword<Linkage>(
                                           parser, LLVM::Linkage::External)));

  // Parse optional visibility, default to Default.
  result.addAttribute(getVisibility_AttrName(result.name),
                      parser.getBuilder().getI64IntegerAttr(
                          parseOptionalLLVMKeyword<LLVM::Visibility, int64_t>(
                              parser, LLVM::Visibility::Default)));

  // Parse optional UnnamedAddr, default to None.
  result.addAttribute(getUnnamedAddrAttrName(result.name),
                      parser.getBuilder().getI64IntegerAttr(
                          parseOptionalLLVMKeyword<UnnamedAddr, int64_t>(
                              parser, LLVM::UnnamedAddr::None)));

  // Default to C Calling Convention if no keyword is provided.
  result.addAttribute(
      getCConvAttrName(result.name),
      CConvAttr::get(parser.getContext(),
                     parseOptionalLLVMKeyword<CConv>(parser, LLVM::CConv::C)));

  StringAttr nameAttr;
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  bool isVariadic;

  auto signatureLocation = parser.getCurrentLocation();
  if (parser.parseSymbolName(nameAttr, getSymNameAttrName(result.name),
                             result.attributes) ||
      function_interface_impl::parseFunctionSignatureWithArguments(
          parser, /*allowVariadic=*/true, entryArgs, isVariadic, resultTypes,
          resultAttrs))
    return failure();

  SmallVector<Type> argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  auto type =
      buildLLVMFunctionType(parser, signatureLocation, argTypes, resultTypes,
                            function_interface_impl::VariadicFlag(isVariadic));
  if (!type)
    return failure();
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));

  if (succeeded(parser.parseOptionalKeyword("vscale_range"))) {
    int64_t minRange, maxRange;
    if (parser.parseLParen() || parser.parseInteger(minRange) ||
        parser.parseComma() || parser.parseInteger(maxRange) ||
        parser.parseRParen())
      return failure();
    auto intTy = IntegerType::get(parser.getContext(), 32);
    result.addAttribute(
        getVscaleRangeAttrName(result.name),
        LLVM::VScaleRangeAttr::get(parser.getContext(),
                                   IntegerAttr::get(intTy, minRange),
                                   IntegerAttr::get(intTy, maxRange)));
  }
  // Parse the optional comdat selector.
  if (succeeded(parser.parseOptionalKeyword("comdat"))) {
    SymbolRefAttr comdat;
    if (parser.parseLParen() || parser.parseAttribute(comdat) ||
        parser.parseRParen())
      return failure();

    result.addAttribute(getComdatAttrName(result.name), comdat);
  }

  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();
  call_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, entryArgs, resultAttrs,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));

  auto *body = result.addRegion();
  OptionalParseResult parseResult =
      parser.parseOptionalRegion(*body, entryArgs);
  return failure(parseResult.has_value() && failed(*parseResult));
}

// Print the LLVMFuncOp. Collects argument and result types and passes them to
// helper functions. Drops "void" result since it cannot be parsed back. Skips
// the external linkage since it is the default value.
void LLVMFuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  if (getLinkage() != LLVM::Linkage::External)
    p << stringifyLinkage(getLinkage()) << ' ';
  StringRef visibility = stringifyVisibility(getVisibility_());
  if (!visibility.empty())
    p << visibility << ' ';
  if (auto unnamedAddr = getUnnamedAddr()) {
    StringRef str = stringifyUnnamedAddr(*unnamedAddr);
    if (!str.empty())
      p << str << ' ';
  }
  if (getCConv() != LLVM::CConv::C)
    p << stringifyCConv(getCConv()) << ' ';

  p.printSymbolName(getName());

  LLVMFunctionType fnType = getFunctionType();
  SmallVector<Type, 8> argTypes;
  SmallVector<Type, 1> resTypes;
  argTypes.reserve(fnType.getNumParams());
  for (unsigned i = 0, e = fnType.getNumParams(); i < e; ++i)
    argTypes.push_back(fnType.getParamType(i));

  Type returnType = fnType.getReturnType();
  if (!llvm::isa<LLVMVoidType>(returnType))
    resTypes.push_back(returnType);

  function_interface_impl::printFunctionSignature(p, *this, argTypes,
                                                  isVarArg(), resTypes);

  // Print vscale range if present
  if (std::optional<VScaleRangeAttr> vscale = getVscaleRange())
    p << " vscale_range(" << vscale->getMinRange().getInt() << ", "
      << vscale->getMaxRange().getInt() << ')';

  // Print the optional comdat selector.
  if (auto comdat = getComdat())
    p << " comdat(" << *comdat << ')';

  function_interface_impl::printFunctionAttributes(
      p, *this,
      {getFunctionTypeAttrName(), getArgAttrsAttrName(), getResAttrsAttrName(),
       getLinkageAttrName(), getCConvAttrName(), getVisibility_AttrName(),
       getComdatAttrName(), getUnnamedAddrAttrName(),
       getVscaleRangeAttrName()});

  // Print the body if this is not an external function.
  Region &body = getBody();
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

LogicalResult LLVMFuncOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyComdat(*this, getComdat(), symbolTable);
}

// Verifies LLVM- and implementation-specific properties of the LLVM func Op:
// - functions don't have 'common' linkage
// - external functions have 'external' or 'extern_weak' linkage;
// - vararg is (currently) only supported for external functions;
LogicalResult LLVMFuncOp::verify() {
  if (getLinkage() == LLVM::Linkage::Common)
    return emitOpError() << "functions cannot have '"
                         << stringifyLinkage(LLVM::Linkage::Common)
                         << "' linkage";

  if (isExternal()) {
    if (getFunctionEntryCountAttr())
      return emitOpError() << "external functions cannot have "
                           << getFunctionEntryCountAttrName() << " attribute";

    if (getLinkage() != LLVM::Linkage::External &&
        getLinkage() != LLVM::Linkage::ExternWeak)
      return emitOpError() << "external functions must have '"
                           << stringifyLinkage(LLVM::Linkage::External)
                           << "' or '"
                           << stringifyLinkage(LLVM::Linkage::ExternWeak)
                           << "' linkage";
    return success();
  }

  // In LLVM IR, these attributes are composed by convention, not by design.
  if (isNoInline() && isAlwaysInline())
    return emitError("no_inline and always_inline attributes are incompatible");

  if (isOptimizeNone() && !isNoInline())
    return emitOpError("with optimize_none must also be no_inline");

  Type landingpadResultTy;
  StringRef diagnosticMessage;
  bool isLandingpadTypeConsistent =
      !walk([&](Operation *op) {
         const auto checkType = [&](Type type, StringRef errorMessage) {
           if (!landingpadResultTy) {
             landingpadResultTy = type;
             return WalkResult::advance();
           }
           if (landingpadResultTy != type) {
             diagnosticMessage = errorMessage;
             return WalkResult::interrupt();
           }
           return WalkResult::advance();
         };
         return TypeSwitch<Operation *, WalkResult>(op)
             .Case([&](LandingpadOp landingpad) {
               constexpr StringLiteral errorMessage =
                   "'llvm.landingpad' should have a consistent result type "
                   "inside a function";
               return checkType(landingpad.getType(), errorMessage);
             })
             .Case([&](ResumeOp resume) {
               constexpr StringLiteral errorMessage =
                   "'llvm.resume' should have a consistent input type inside a "
                   "function";
               return checkType(resume.getValue().getType(), errorMessage);
             })
             .Default([](auto) { return WalkResult::skip(); });
       }).wasInterrupted();
  if (!isLandingpadTypeConsistent) {
    assert(!diagnosticMessage.empty() &&
           "Expecting a non-empty diagnostic message");
    return emitError(diagnosticMessage);
  }

  if (failed(verifyBlockTags(*this)))
    return failure();

  return success();
}

/// Verifies LLVM- and implementation-specific properties of the LLVM func Op:
/// - entry block arguments are of LLVM types.
LogicalResult LLVMFuncOp::verifyRegions() {
  if (isExternal())
    return success();

  unsigned numArguments = getFunctionType().getNumParams();
  Block &entryBlock = front();
  for (unsigned i = 0; i < numArguments; ++i) {
    Type argType = entryBlock.getArgument(i).getType();
    if (!isCompatibleType(argType))
      return emitOpError("entry block argument #")
             << i << " is not of LLVM type";
  }

  return success();
}

Region *LLVMFuncOp::getCallableRegion() {
  if (isExternal())
    return nullptr;
  return &getBody();
}

//===----------------------------------------------------------------------===//
// LinkerOptionsOp
//===----------------------------------------------------------------------===//

LogicalResult LinkerOptionsOp::verify() {
  if (mlir::Operation *parentOp = (*this)->getParentOp();
      parentOp && !satisfiesLLVMModule(parentOp))
    return emitOpError("must appear at the module level");
  return success();
}

//===----------------------------------------------------------------------===//
// ModuleFlagsOp
//===----------------------------------------------------------------------===//

LogicalResult ModuleFlagsOp::verify() {
  if (Operation *parentOp = (*this)->getParentOp();
      parentOp && !satisfiesLLVMModule(parentOp))
    return emitOpError("must appear at the module level");

  llvm::DenseSet<StringAttr> seenNonRequireKeys;
  for (Attribute flag : getFlags()) {
    auto moduleFlag = dyn_cast<ModuleFlagAttrInterface>(flag);
    if (!moduleFlag)
      return emitOpError("expected a module flag attribute");
    if (failed(LLVM::detail::verifyModuleFlagValue(
            moduleFlag.getModuleFlagKey(), moduleFlag.getModuleFlagValue(),
            [&] { return emitOpError(); })))
      return failure();
    if (moduleFlag.getModuleFlagBehavior() == ModFlagBehavior::Require)
      continue;
    StringAttr key = moduleFlag.getModuleFlagKey();
    if (!seenNonRequireKeys.insert(key).second)
      return emitOpError("expected module flag key '")
             << key.getValue() << "' to be unique for non-require flags";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

Value mlir::LLVM::createGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     LLVM::Linkage linkage) {
  assert(builder.getInsertionBlock() &&
         builder.getInsertionBlock()->getParentOp() &&
         "expected builder to point to a block constrained in an op");
  auto module =
      builder.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  assert(module && "builder points to an op outside of a module");

  // Create the global at the entry of the module.
  OpBuilder moduleBuilder(module.getBodyRegion(), builder.getListener());
  MLIRContext *ctx = builder.getContext();
  auto type = LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8), value.size());
  auto global = LLVM::GlobalOp::create(
      moduleBuilder, loc, type, /*isConstant=*/true, linkage, name,
      builder.getStringAttr(value), /*alignment=*/0);

  LLVMPointerType ptrType = LLVMPointerType::get(ctx);
  // Get the pointer to the first character in the global string.
  Value globalPtr =
      LLVM::AddressOfOp::create(builder, loc, ptrType, global.getSymNameAttr());
  return LLVM::GEPOp::create(builder, loc, ptrType, type, globalPtr,
                             ArrayRef<GEPArg>{0, 0});
}

bool mlir::LLVM::satisfiesLLVMModule(Operation *op) {
  return op->hasTrait<OpTrait::SymbolTable>() &&
         op->hasTrait<OpTrait::IsIsolatedFromAbove>();
}

Operation *mlir::LLVM::parentLLVMModule(Operation *op) {
  Operation *module = op->getParentOp();
  while (module && !satisfiesLLVMModule(module))
    module = module->getParentOp();
  assert(module && "unexpected operation outside of a module");
  return module;
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMModuleOps.cpp.inc"
