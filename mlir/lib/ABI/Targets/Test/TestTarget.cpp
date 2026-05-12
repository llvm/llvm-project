//===- TestTarget.cpp - Predictable test ABI target ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// **NOT A REAL ABI TARGET.**
//
// This file implements a predictable, dialect-agnostic ABI classifier for
// testing the MLIR ABIRewriteContext infrastructure.  The rules approximate
// x86_64 SysV thresholds (Direct / Extend / Indirect / Ignore / Expand) so
// the generated classifications are familiar to reviewers, but they are
// NOT a substitute for testing against the real x86_64 ABIInfo.  Real
// ABI targets live alongside the LLVM ABI library in `llvm/lib/ABI/Targets/`.
//
// Real-ABI-shaped tests use the classification-injection driver via
// `parseClassificationAttr`, which lets tests construct any
// FunctionClassification (including shapes the test target itself does
// not produce) by attaching a DictionaryAttr to the function.
//
// Rules:
//   - mlir::NoneType                           → Ignore
//   - IntegerType with width < 32              → Extend (zero-extend by
//                                                default; tests using the
//                                                injection driver can
//                                                override to signed)
//   - IntegerType with width >= 32             → Direct
//   - FloatType, VectorType, MemRefType        → Direct
//   - Anything else with DataLayout size 0     → Ignore
//   - Anything else with DataLayout size <= 16 → Direct (coerced to the
//                                                same type — no actual
//                                                coercion in the test
//                                                target; PR C handles
//                                                non-trivial coercion)
//   - Anything else with DataLayout size > 16  → Indirect with byval=true
//                                                (sret on returns) and
//                                                alignment from
//                                                DataLayout
//
//===----------------------------------------------------------------------===//

#include "mlir/ABI/Targets/Test/TestTarget.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Alignment.h"

using namespace mlir;
using namespace mlir::abi;
using namespace mlir::abi::test;

namespace {

/// Indirect-vs-direct cutoff in bytes.  Chosen to match x86_64 SysV's
/// 16-byte register-passing window for reviewer familiarity.
constexpr uint64_t IndirectCutoffBytes = 16;

/// Below this width (in bits) integers get an extension attribute.
/// Chosen to match x86_64 SysV (32-bit register width) for reviewer
/// familiarity.
constexpr unsigned ExtendBelowBits = 32;

ArgClassification classifyOne(Type type, const DataLayout &dl) {
  if (isa<NoneType>(type))
    return ArgClassification::getIgnore();

  if (auto intTy = dyn_cast<IntegerType>(type)) {
    if (intTy.getWidth() < ExtendBelowBits) {
      Type i32Ty = IntegerType::get(type.getContext(), ExtendBelowBits);
      return ArgClassification::getExtend(i32Ty, /*signExt=*/intTy.isSigned());
    }
    return ArgClassification::getDirect();
  }

  if (auto indexTy = dyn_cast<IndexType>(type)) {
    llvm::TypeSize sizeInBits = dl.getTypeSizeInBits(indexTy);
    if (sizeInBits.getFixedValue() < ExtendBelowBits) {
      Type i32Ty = IntegerType::get(type.getContext(), ExtendBelowBits);
      return ArgClassification::getExtend(i32Ty, /*signExt=*/true);
    }
    return ArgClassification::getDirect();
  }

  if (isa<FloatType, VectorType, MemRefType>(type))
    return ArgClassification::getDirect();

  // For dialect-specific types: query DataLayout via
  // DataLayoutTypeInterface.  Types that don't implement the interface
  // (e.g. dialect-specific void / unit-style sentinel types used as a
  // function's "no return value" marker) are treated as Ignore so that
  // the test target degrades gracefully rather than crashing on unknown
  // types.
  if (!isa<DataLayoutTypeInterface>(type))
    return ArgClassification::getIgnore();

  llvm::TypeSize sizeInBits = dl.getTypeSizeInBits(type);
  if (sizeInBits.isZero())
    return ArgClassification::getIgnore();

  uint64_t sizeInBytes = (sizeInBits.getFixedValue() + 7) / 8;
  if (sizeInBytes <= IndirectCutoffBytes)
    return ArgClassification::getDirect();

  uint64_t alignBytes = dl.getTypeABIAlignment(type);
  return ArgClassification::getIndirect(llvm::Align(alignBytes),
                                        /*byVal=*/true);
}

} // namespace

FunctionClassification mlir::abi::test::classify(ArrayRef<Type> argTypes,
                                                 Type returnType,
                                                 const DataLayout &dl) {
  FunctionClassification fc;
  fc.returnInfo = classifyOne(returnType, dl);
  fc.argInfos.reserve(argTypes.size());
  for (Type t : argTypes)
    fc.argInfos.push_back(classifyOne(t, dl));
  return fc;
}

namespace {

/// Set of dictionary keys this parser knows about.  Any key not in this
/// set causes a parse error (no silent ignore).  Updated when new
/// optional keys are added to the schema.
constexpr StringRef knownArgKeys[] = {
    "kind",        "coerced_type",   "sign_extend",
    "can_flatten", "indirect_align", "byval",
};

bool isKnownArgKey(StringRef key) {
  for (StringRef k : knownArgKeys)
    if (k == key)
      return true;
  return false;
}

/// Parse a single ArgClassification dictionary.  Returns std::nullopt on
/// any error (with the diagnostic emitted via \p emitError).
std::optional<ArgClassification>
parseOne(DictionaryAttr argDict, function_ref<InFlightDiagnostic()> emitError) {
  StringAttr kindAttr = argDict.getAs<StringAttr>("kind");
  if (!kindAttr) {
    emitError() << "missing required 'kind' StringAttr";
    return std::nullopt;
  }

  for (NamedAttribute na : argDict)
    if (!isKnownArgKey(na.getName().getValue())) {
      emitError() << "unknown key '" << na.getName().getValue()
                  << "' in classification dictionary; allowed keys are "
                  << "kind, coerced_type, sign_extend, can_flatten, "
                  << "indirect_align, byval";
      return std::nullopt;
    }

  StringRef kind = kindAttr.getValue();

  if (kind == "direct") {
    Type coerced;
    if (auto t = argDict.getAs<TypeAttr>("coerced_type"))
      coerced = t.getValue();
    auto c = ArgClassification::getDirect(coerced);
    if (auto cf = argDict.getAs<BoolAttr>("can_flatten"))
      c.canFlatten = cf.getValue();
    return c;
  }

  if (kind == "extend") {
    auto coerced = argDict.getAs<TypeAttr>("coerced_type");
    if (!coerced) {
      emitError() << "kind='extend' requires 'coerced_type' TypeAttr";
      return std::nullopt;
    }
    bool signExt = false;
    if (auto se = argDict.getAs<BoolAttr>("sign_extend"))
      signExt = se.getValue();
    return ArgClassification::getExtend(coerced.getValue(), signExt);
  }

  if (kind == "indirect") {
    auto align = argDict.getAs<IntegerAttr>("indirect_align");
    if (!align) {
      emitError() << "kind='indirect' requires 'indirect_align' IntegerAttr";
      return std::nullopt;
    }
    if (align.getInt() <= 0 || !llvm::isPowerOf2_64(align.getInt())) {
      emitError() << "'indirect_align' must be a positive power of 2; got "
                  << align.getInt();
      return std::nullopt;
    }
    bool byVal = true;
    if (auto bv = argDict.getAs<BoolAttr>("byval"))
      byVal = bv.getValue();
    return ArgClassification::getIndirect(llvm::Align(align.getInt()), byVal);
  }

  if (kind == "ignore") {
    return ArgClassification::getIgnore();
  }

  if (kind == "expand") {
    ArgClassification c;
    c.kind = ArgKind::Expand;
    return c;
  }

  emitError() << "unknown kind='" << kind
              << "'; expected one of direct, extend, indirect, ignore, expand";
  return std::nullopt;
}

} // namespace

std::optional<FunctionClassification> mlir::abi::test::parseClassificationAttr(
    DictionaryAttr attr, function_ref<InFlightDiagnostic()> emitError) {
  auto returnDict = attr.getAs<DictionaryAttr>("return");
  if (!returnDict) {
    emitError() << "missing required 'return' DictionaryAttr";
    return std::nullopt;
  }

  auto argsArr = attr.getAs<ArrayAttr>("args");
  if (!argsArr) {
    emitError() << "missing required 'args' ArrayAttr";
    return std::nullopt;
  }

  for (NamedAttribute na : attr) {
    StringRef k = na.getName().getValue();
    if (k != "return" && k != "args") {
      emitError() << "unknown top-level key '" << k
                  << "'; only 'return' and 'args' are allowed";
      return std::nullopt;
    }
  }

  FunctionClassification fc;

  std::optional<ArgClassification> ret = parseOne(returnDict, emitError);
  if (!ret)
    return std::nullopt;
  fc.returnInfo = *ret;

  fc.argInfos.reserve(argsArr.size());
  for (Attribute a : argsArr) {
    auto d = dyn_cast<DictionaryAttr>(a);
    if (!d) {
      emitError() << "'args' entries must be DictionaryAttrs";
      return std::nullopt;
    }
    std::optional<ArgClassification> ac = parseOne(d, emitError);
    if (!ac)
      return std::nullopt;
    fc.argInfos.push_back(*ac);
  }

  return fc;
}
