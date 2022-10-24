//===- LLVMAttrs.cpp - LLVM Attributes registration -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the attribute details for the LLVM IR dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"

using namespace mlir;
using namespace mlir::LLVM;

#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// LLVMDialect registration
//===----------------------------------------------------------------------===//

void LLVMDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DINodeAttr
//===----------------------------------------------------------------------===//

bool DINodeAttr::classof(Attribute attr) {
  return llvm::isa<DIBasicTypeAttr, DICompileUnitAttr, DICompositeTypeAttr,
                   DIDerivedTypeAttr, DIFileAttr, DILexicalBlockAttr,
                   DILexicalBlockFileAttr, DILocalVariableAttr,
                   DISubprogramAttr, DISubroutineTypeAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DIScopeAttr
//===----------------------------------------------------------------------===//

bool DIScopeAttr::classof(Attribute attr) {
  return llvm::isa<DICompileUnitAttr, DIFileAttr, DILexicalBlockAttr,
                   DILexicalBlockFileAttr, DISubprogramAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DITypeAttr
//===----------------------------------------------------------------------===//

bool DITypeAttr::classof(Attribute attr) {
  return llvm::isa<DIBasicTypeAttr, DISubroutineTypeAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DICompileUnitAttr
//===----------------------------------------------------------------------===//

void DICompileUnitAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getFile());
  walkAttrsFn(getProducer());
}

Attribute
DICompileUnitAttr::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                               ArrayRef<Type> replTypes) const {
  return get(getContext(), getSourceLanguage(), replAttrs[0].cast<DIFileAttr>(),
             replAttrs[1].cast<StringAttr>(), getIsOptimized(),
             getEmissionKind());
}

//===----------------------------------------------------------------------===//
// DICompositeTypeAttr
//===----------------------------------------------------------------------===//

void DICompositeTypeAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getName());
  walkAttrsFn(getFile());
  walkAttrsFn(getScope());
  for (DINodeAttr element : getElements())
    walkAttrsFn(element);
}

Attribute DICompositeTypeAttr::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  ArrayRef<Attribute> elements = replAttrs.drop_front(3);
  return get(
      getContext(), getTag(), replAttrs[0].cast<StringAttr>(),
      cast_or_null<DIFileAttr>(replAttrs[1]), getLine(),
      cast_or_null<DIScopeAttr>(replAttrs[2]), getSizeInBits(),
      getAlignInBits(),
      ArrayRef<DINodeAttr>(static_cast<const DINodeAttr *>(elements.data()),
                           elements.size()));
}

//===----------------------------------------------------------------------===//
// DIDerivedTypeAttr
//===----------------------------------------------------------------------===//

void DIDerivedTypeAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getName());
  walkAttrsFn(getBaseType());
}

Attribute
DIDerivedTypeAttr::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                               ArrayRef<Type> replTypes) const {
  return get(getContext(), getTag(), replAttrs[0].cast<StringAttr>(),
             replAttrs[1].cast<DITypeAttr>(), getSizeInBits(), getAlignInBits(),
             getOffsetInBits());
}

//===----------------------------------------------------------------------===//
// DILexicalBlockAttr
//===----------------------------------------------------------------------===//

void DILexicalBlockAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getScope());
  walkAttrsFn(getFile());
}

Attribute DILexicalBlockAttr::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  return get(replAttrs[0].cast<DIScopeAttr>(), replAttrs[1].cast<DIFileAttr>(),
             getLine(), getColumn());
}

//===----------------------------------------------------------------------===//
// DILexicalBlockFileAttr
//===----------------------------------------------------------------------===//

void DILexicalBlockFileAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getScope());
  walkAttrsFn(getFile());
}

Attribute DILexicalBlockFileAttr::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  return get(replAttrs[0].cast<DIScopeAttr>(), replAttrs[1].cast<DIFileAttr>(),
             getDescriminator());
}

//===----------------------------------------------------------------------===//
// DILocalVariableAttr
//===----------------------------------------------------------------------===//

void DILocalVariableAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getScope());
  walkAttrsFn(getName());
  walkAttrsFn(getFile());
  walkAttrsFn(getType());
}

Attribute DILocalVariableAttr::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  return get(getContext(), replAttrs[0].cast<DIScopeAttr>(),
             replAttrs[1].cast<StringAttr>(), replAttrs[2].cast<DIFileAttr>(),
             getLine(), getArg(), getAlignInBits(),
             replAttrs[3].cast<DITypeAttr>());
}

//===----------------------------------------------------------------------===//
// DISubprogramAttr
//===----------------------------------------------------------------------===//

void DISubprogramAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getCompileUnit());
  walkAttrsFn(getScope());
  walkAttrsFn(getName());
  walkAttrsFn(getLinkageName());
  walkAttrsFn(getFile());
  walkAttrsFn(getType());
}

Attribute
DISubprogramAttr::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                              ArrayRef<Type> replTypes) const {
  return get(getContext(), replAttrs[0].cast<DICompileUnitAttr>(),
             replAttrs[1].cast<DIScopeAttr>(), replAttrs[2].cast<StringAttr>(),
             replAttrs[3].cast<StringAttr>(), replAttrs[4].cast<DIFileAttr>(),
             getLine(), getScopeLine(), getSubprogramFlags(),
             replAttrs[5].cast<DISubroutineTypeAttr>());
}

//===----------------------------------------------------------------------===//
// DISubroutineTypeAttr
//===----------------------------------------------------------------------===//

void DISubroutineTypeAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  for (DITypeAttr type : getTypes())
    walkAttrsFn(type);
}

Attribute DISubroutineTypeAttr::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  return get(
      getContext(), getCallingConvention(),
      ArrayRef<DITypeAttr>(static_cast<const DITypeAttr *>(replAttrs.data()),
                           replAttrs.size()));
}

//===----------------------------------------------------------------------===//
// LoopOptionsAttrBuilder
//===----------------------------------------------------------------------===//

LoopOptionsAttrBuilder::LoopOptionsAttrBuilder(LoopOptionsAttr attr)
    : options(attr.getOptions().begin(), attr.getOptions().end()) {}

template <typename T>
LoopOptionsAttrBuilder &LoopOptionsAttrBuilder::setOption(LoopOptionCase tag,
                                                          Optional<T> value) {
  auto option = llvm::find_if(
      options, [tag](auto option) { return option.first == tag; });
  if (option != options.end()) {
    if (value)
      option->second = *value;
    else
      options.erase(option);
  } else {
    options.push_back(LoopOptionsAttr::OptionValuePair(tag, *value));
  }
  return *this;
}

LoopOptionsAttrBuilder &
LoopOptionsAttrBuilder::setDisableLICM(Optional<bool> value) {
  return setOption(LoopOptionCase::disable_licm, value);
}

/// Set the `interleave_count` option to the provided value. If no value
/// is provided the option is deleted.
LoopOptionsAttrBuilder &
LoopOptionsAttrBuilder::setInterleaveCount(Optional<uint64_t> count) {
  return setOption(LoopOptionCase::interleave_count, count);
}

/// Set the `disable_unroll` option to the provided value. If no value
/// is provided the option is deleted.
LoopOptionsAttrBuilder &
LoopOptionsAttrBuilder::setDisableUnroll(Optional<bool> value) {
  return setOption(LoopOptionCase::disable_unroll, value);
}

/// Set the `disable_pipeline` option to the provided value. If no value
/// is provided the option is deleted.
LoopOptionsAttrBuilder &
LoopOptionsAttrBuilder::setDisablePipeline(Optional<bool> value) {
  return setOption(LoopOptionCase::disable_pipeline, value);
}

/// Set the `pipeline_initiation_interval` option to the provided value.
/// If no value is provided the option is deleted.
LoopOptionsAttrBuilder &LoopOptionsAttrBuilder::setPipelineInitiationInterval(
    Optional<uint64_t> count) {
  return setOption(LoopOptionCase::pipeline_initiation_interval, count);
}

//===----------------------------------------------------------------------===//
// LoopOptionsAttr
//===----------------------------------------------------------------------===//

template <typename T>
static Optional<T>
getOption(ArrayRef<std::pair<LoopOptionCase, int64_t>> options,
          LoopOptionCase option) {
  auto it =
      lower_bound(options, option, [](auto optionPair, LoopOptionCase option) {
        return optionPair.first < option;
      });
  if (it == options.end())
    return {};
  return static_cast<T>(it->second);
}

Optional<bool> LoopOptionsAttr::disableUnroll() {
  return getOption<bool>(getOptions(), LoopOptionCase::disable_unroll);
}

Optional<bool> LoopOptionsAttr::disableLICM() {
  return getOption<bool>(getOptions(), LoopOptionCase::disable_licm);
}

Optional<int64_t> LoopOptionsAttr::interleaveCount() {
  return getOption<int64_t>(getOptions(), LoopOptionCase::interleave_count);
}

/// Build the LoopOptions Attribute from a sorted array of individual options.
LoopOptionsAttr LoopOptionsAttr::get(
    MLIRContext *context,
    ArrayRef<std::pair<LoopOptionCase, int64_t>> sortedOptions) {
  assert(llvm::is_sorted(sortedOptions, llvm::less_first()) &&
         "LoopOptionsAttr ctor expects a sorted options array");
  return Base::get(context, sortedOptions);
}

/// Build the LoopOptions Attribute from a sorted array of individual options.
LoopOptionsAttr LoopOptionsAttr::get(MLIRContext *context,
                                     LoopOptionsAttrBuilder &optionBuilders) {
  llvm::sort(optionBuilders.options, llvm::less_first());
  return Base::get(context, optionBuilders.options);
}

void LoopOptionsAttr::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::interleaveComma(getOptions(), printer, [&](auto option) {
    printer << stringifyEnum(option.first) << " = ";
    switch (option.first) {
    case LoopOptionCase::disable_licm:
    case LoopOptionCase::disable_unroll:
    case LoopOptionCase::disable_pipeline:
      printer << (option.second ? "true" : "false");
      break;
    case LoopOptionCase::interleave_count:
    case LoopOptionCase::pipeline_initiation_interval:
      printer << option.second;
      break;
    }
  });
  printer << ">";
}

Attribute LoopOptionsAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};

  SmallVector<std::pair<LoopOptionCase, int64_t>> options;
  llvm::SmallDenseSet<LoopOptionCase> seenOptions;
  auto parseLoopOptions = [&]() -> ParseResult {
    StringRef optionName;
    if (parser.parseKeyword(&optionName))
      return failure();

    auto option = symbolizeLoopOptionCase(optionName);
    if (!option)
      return parser.emitError(parser.getNameLoc(), "unknown loop option: ")
             << optionName;
    if (!seenOptions.insert(*option).second)
      return parser.emitError(parser.getNameLoc(), "loop option present twice");
    if (failed(parser.parseEqual()))
      return failure();

    int64_t value;
    switch (*option) {
    case LoopOptionCase::disable_licm:
    case LoopOptionCase::disable_unroll:
    case LoopOptionCase::disable_pipeline:
      if (succeeded(parser.parseOptionalKeyword("true")))
        value = 1;
      else if (succeeded(parser.parseOptionalKeyword("false")))
        value = 0;
      else {
        return parser.emitError(parser.getNameLoc(),
                                "expected boolean value 'true' or 'false'");
      }
      break;
    case LoopOptionCase::interleave_count:
    case LoopOptionCase::pipeline_initiation_interval:
      if (failed(parser.parseInteger(value)))
        return parser.emitError(parser.getNameLoc(), "expected integer value");
      break;
    }
    options.push_back(std::make_pair(*option, value));
    return success();
  };
  if (parser.parseCommaSeparatedList(parseLoopOptions) || parser.parseGreater())
    return {};

  llvm::sort(options, llvm::less_first());
  return get(parser.getContext(), options);
}
